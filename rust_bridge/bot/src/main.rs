// buta-ppo/rust_bridge/bot/src/main.rs
mod bot_ffi;
mod self_test;

use std::ffi::CString;
use std::os::raw::c_void;
use std::sync::{Arc, OnceLock};

use rust_bridge::{CArenaState, CBallState, CCarControls, CCarState, CPhysState};
use crate::bot_ffi::{
    bot_api_free, bot_api_get_action, bot_api_init, bot_api_max_players_per_team, bot_api_tick_skip,
};

use rlbot_rocketsim::rlbot::agents::{run_bot_agents, BotAgent};
use rlbot_rocketsim::rlbot::flat::{
    ControllableInfo, FieldInfo, GamePacket, Human, MatchConfiguration, MatchPhase, PlayerClass, PlayerConfiguration,
    PlayerInput,
};
use rlbot_rocketsim::rlbot::util::{AgentEnvironment, PacketQueue};
use rlbot_rocketsim::rlbot::RLBotConnection;
use rlbot_rocketsim::{EnrichmentError, GameStateEnricher, MatchContext, MatchContextError};

use rlbot_rocketsim::rlbot::flat::ControllerState;

/// Maps C++ CCarControls into the RLBot ControllerState flatbuffer.
fn to_rlbot_controls(controls: CCarControls) -> ControllerState {
    ControllerState {
        throttle: controls.throttle,
        steer: controls.steer,
        pitch: controls.pitch,
        yaw: controls.yaw,
        roll: controls.roll,
        jump: controls.jump,
        boost: controls.boost,
        handbrake: controls.handbrake,
        use_item: false,
    }
}

/// The main bot instance that holds the C++ pointer and simulation enricher.
pub struct ButaPPOBot {
    player_index: usize,
    enricher: GameStateEnricher,
    ctx_ptr: *mut c_void,
    held_controls: CCarControls,
    next_decision_frame: Option<u32>,
    tick_skip: u32,
    big_pad_indices: Vec<usize>,
    small_pad_indices: Vec<usize>,
    last_phase: Option<MatchPhase>,
    pending_reshuffle: bool,
    // The match's players, plus any who joined after the bot started, which the enricher is rebuilt from.
    match_config: MatchConfiguration,
    field_info: Arc<FieldInfo>,
    max_per_team: usize,
    // Why the bot isn't inferring, while it holds neutral controls instead. Printed when it changes.
    pause_reason: Option<String>,
}

impl BotAgent for ButaPPOBot {
    fn new(
        _team: u32,
        controllable_info: ControllableInfo,
        match_config: Arc<MatchConfiguration>,
        field_info: Arc<FieldInfo>,
        _packet_queue: &mut PacketQueue,
    ) -> Self {
        let path_str = MODEL_PATH.get().expect("model path set in main");
        let path_c = CString::new(path_str.as_str()).expect("Invalid CString");
        let meshes_c = CString::new(COLLISION_MESHES).expect("Invalid CString");
        let ctx_ptr = unsafe { bot_api_init(path_c.as_ptr(), meshes_c.as_ptr()) };

        if ctx_ptr.is_null() {
            panic!("Failed to load C++ LibTorch checkpoint at: {}", path_str);
        }

        // The obs has a fixed number of car slots, so exit if a team won't fit.
        let max_per_team = unsafe { bot_api_max_players_per_team(ctx_ptr) } as usize;
        let players = &match_config.player_configurations;
        let blue_count = players.iter().filter(|p| p.team == 0).count();
        let orange_count = players.len() - blue_count;

        if blue_count > max_per_team || orange_count > max_per_team {
            eprintln!(
                "[buta_ppo_bot] Invalid team layout: {}v{}, but the model supports at most {} players per team. Exiting.",
                blue_count, orange_count, max_per_team
            );
            std::process::exit(1);
        }

        let context = MatchContext::new(&match_config, &field_info)
            .expect("create RocketSim context from RLBot match");

        // Sort the boost pads based on RLBot's static field info.
        let mut big_pads = Vec::new();
        let mut small_pads = Vec::new();

        for (i, pad) in field_info.boost_pads.iter().enumerate() {
            let loc = &pad.location;
            if pad.is_full_boost {
                big_pads.push((i, loc.y, loc.x));
            } else {
                small_pads.push((i, loc.y, loc.x));
            }
        }

        // Sort by Y first, then X.
        big_pads.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.total_cmp(&b.2)));
        small_pads.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.total_cmp(&b.2)));

        let big_pad_indices: Vec<usize> = big_pads.into_iter().map(|(i, _, _)| i).collect();
        let small_pad_indices: Vec<usize> = small_pads.into_iter().map(|(i, _, _)| i).collect();

        Self {
            player_index: controllable_info.index as usize,
            enricher: GameStateEnricher::from_match_context(context),
            ctx_ptr,
            held_controls: CCarControls::default(),
            next_decision_frame: None,
            tick_skip: unsafe { bot_api_tick_skip(ctx_ptr) },
            big_pad_indices,
            small_pad_indices,
            last_phase: None,
            pending_reshuffle: false,
            match_config: (*match_config).clone(),
            field_info,
            max_per_team,
            pause_reason: None,
        }
    }

    fn tick(&mut self, packet: &GamePacket, packet_queue: &mut PacketQueue) {
        let frame = packet.match_info.frame_num;
        let phase = packet.match_info.match_phase;

        // Flag a reshuffle on entering a kickoff; consumed by the next infer.
        if matches!(phase, MatchPhase::Kickoff) && !matches!(self.last_phase, Some(MatchPhase::Kickoff)) {
            self.pending_reshuffle = true;
        }
        self.last_phase = Some(phase);

        // Feed the RLBot packet into the RocketSim enricher.
        let ready = self.ready(packet);

        // Only infer new actions during Kickoff or Active gameplay.
        if ready
            && matches!(
                packet.match_info.match_phase,
                MatchPhase::Kickoff | MatchPhase::Active
            )
            && packet.players.get(self.player_index).is_some()
            && self.next_decision_frame.is_none_or(|next_frame| frame >= next_frame)
        {
            self.infer(packet, frame);
        }

        // Send the current controls back to the RLBot server.
        packet_queue.push(PlayerInput {
            player_index: self.player_index as u32,
            controller_state: to_rlbot_controls(self.held_controls),
        });
    }
}

impl ButaPPOBot {
    /// Enriches the packet and checks the model can play it. Otherwise the bot holds neutral controls, rather than
    /// the last ones it chose, until it can.
    fn ready(&mut self, packet: &GamePacket) -> bool {
        let blue_count = packet.players.iter().filter(|p| p.team == 0).count();
        let orange_count = packet.players.len() - blue_count;

        let problem = match self.enrich(packet) {
            Err(e) => Some(format!("Couldn't mirror the game in RocketSim: {e}")),
            // The obs has a fixed number of car slots, so a team past it can't be played until a player leaves.
            Ok(()) if blue_count > self.max_per_team || orange_count > self.max_per_team => Some(format!(
                "{}v{} on the field, but the model supports at most {} players per team",
                blue_count, orange_count, self.max_per_team
            )),
            Ok(()) => None,
        };

        // Reported only in play, so packets between plays can't flip it back and forth.
        let in_play = matches!(packet.match_info.match_phase, MatchPhase::Kickoff | MatchPhase::Active);
        match problem {
            Some(reason) => {
                if in_play && self.pause_reason.as_ref() != Some(&reason) {
                    eprintln!("[buta_ppo_bot] {reason}. Holding neutral controls.");
                    self.pause_reason = Some(reason);
                }
                self.held_controls = CCarControls::default();
                self.next_decision_frame = None;
                false
            }
            None => {
                if in_play && self.pause_reason.take().is_some() {
                    eprintln!("[buta_ppo_bot] Playing again.");
                }
                true
            }
        }
    }

    /// Feeds the packet into the enricher, first adding any players who joined after the bot started.
    fn enrich(&mut self, packet: &GamePacket) -> Result<(), EnrichmentError> {
        match self.enricher.update(packet) {
            Err(EnrichmentError::MatchContext(MatchContextError::PlayerNotConfigured { .. })) => {
                self.add_joined_players(packet)?;
                self.enricher.update(packet).map(|_| ())
            }
            result => result.map(|_| ()),
        }
    }

    /// Adds the packet's players missing from the match config, as humans (whose hitbox the enricher reads from the
    /// packet), and rebuilds the enricher with them. The enricher rejects every packet holding a player it wasn't built
    /// with.
    fn add_joined_players(&mut self, packet: &GamePacket) -> Result<(), MatchContextError> {
        for player in &packet.players {
            if !self.match_config.player_configurations.iter().any(|p| p.player_id == player.player_id) {
                eprintln!("[buta_ppo_bot] {} joined team {} mid-match.", player.name, player.team);
                self.match_config.player_configurations.push(PlayerConfiguration {
                    variety: PlayerClass::Human(Box::new(Human {})),
                    team: player.team,
                    player_id: player.player_id,
                });
            }
        }
        let context = MatchContext::new(&self.match_config, &self.field_info)?;
        self.enricher = GameStateEnricher::from_match_context(context);
        Ok(())
    }

    fn infer(&mut self, packet: &GamePacket, frame: u32) {
        // Construct the FFI C++ Arena State
        let mut c_arena = CArenaState::default();
        c_arena.tick_count = frame as u64;

        // Pack the Ball.
        if let Some(ball) = packet.balls.first() {
            let phys = &ball.physics;
            c_arena.ball = CBallState {
                phys: CPhysState {
                    pos: [phys.location.x, phys.location.y, phys.location.z],
                    vel: [phys.velocity.x, phys.velocity.y, phys.velocity.z],
                    ang_vel: [phys.angular_velocity.x, phys.angular_velocity.y, phys.angular_velocity.z],
                    rot_mat: [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], 
                }
            };
        }

        // Pack the Cars.
        let num_cars = packet.players.len().min(8);
        c_arena.num_cars = num_cars as u32;
        let mut my_car_idx = 0;

        for i in 0..num_cars {
            let player = &packet.players[i];
            
            // Check if this slot belongs to our bot
            if i == self.player_index {
                my_car_idx = i as u32;
            }

            // Extract the enriched state using the stable player_id
            if let Some(rs_car_state) = self.enricher.car_state_by_player_id(player.player_id) {
                c_arena.cars[i] = CCarState {
                    phys: CPhysState::from(&rs_car_state.phys),
                    team: player.team as i32,
                    boost: rs_car_state.boost,
                    handbrake_val: rs_car_state.handbrake_val,
                    air_time_since_jump: rs_car_state.air_time_since_jump,
                    demo_respawn_timer: rs_car_state.demo_respawn_timer,
                    is_on_ground: rs_car_state.is_on_ground(),
                    has_jumped: rs_car_state.has_jumped,
                    has_double_jumped: rs_car_state.has_double_jumped,
                    is_jumping: rs_car_state.is_jumping,
                    has_flip_or_jump: rs_car_state.has_flip_or_jump(),
                    has_flipped: rs_car_state.has_flipped,
                    is_auto_flipping: rs_car_state.is_auto_flipping,
                    is_flipping: rs_car_state.is_flipping,
                    is_demoed: rs_car_state.is_demoed,
                    is_supersonic: rs_car_state.is_supersonic,
                };
            }
        }

        // Pack the Boost Pads.
        for (i, &idx) in self.big_pad_indices.iter().enumerate() {
            c_arena.boost_pads.big[i] = packet.boost_pads[idx].timer.max(0.0);
        }
        for (i, &idx) in self.small_pad_indices.iter().enumerate() {
            c_arena.boost_pads.small[i] = packet.boost_pads[idx].timer.max(0.0);
        }

        // Execute LibTorch Forward Pass.
        self.held_controls = unsafe { bot_api_get_action(self.ctx_ptr, &c_arena, my_car_idx, self.pending_reshuffle) };
        self.pending_reshuffle = false;
        self.next_decision_frame = Some(frame + self.tick_skip);
    }
}

impl Drop for ButaPPOBot {
    fn drop(&mut self) {
        unsafe { bot_api_free(self.ctx_ptr) };
    }
}

/// The model to run, read from the command line in main.
static MODEL_PATH: OnceLock<String> = OnceLock::new();

/// The collision meshes, relative to the working directory (RLBot runs the bot in its directory).
pub const COLLISION_MESHES: &str = "collision_meshes";

/// Loads the C++ library from libs/ next to the executable, with its dependencies from the same directory. Windows has
/// no rpath, so the binary delay-loads the library (see build.rs) and loads it here before its first call, which then
/// finds it already loaded.
#[cfg(windows)]
fn load_bot_library() -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(file_name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    }
    // Search the loaded DLL's own directory for its dependencies, then the usual system directories.
    const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x0000_0100;
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;

    let exe = std::env::current_exe().map_err(|e| format!("Couldn't find the bot executable's path: {e}"))?;
    let path = exe.with_file_name("libs").join("buta_ppo_bot.dll");
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    if unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), flags) }.is_null() {
        return Err(format!(
            "Couldn't load {}: {}. It or a library it needs is missing from libs/; run deploy.py again.",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

fn main() {
    // Usage: buta_ppo_bot [--self-test] [model]. The model is relative to the working directory (RLBot runs the bot in
    // its directory) and defaults to model.pt. See self_test.rs for the self-test's model.
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let self_test = args.first().is_some_and(|arg| arg == "--self-test");
    if self_test {
        args.remove(0);
    }
    let model_arg = args.into_iter().next();
    MODEL_PATH.set(model_arg.clone().unwrap_or_else(|| "model.pt".into())).expect("set once");

    #[cfg(windows)]
    if let Err(message) = load_bot_library() {
        eprintln!("[buta_ppo_bot] {message}");
        std::process::exit(1);
    }

    // For this binary's RocketSim, which mirrors the game for the obs. The C++ library loads its own (bot_api_init).
    rocketsim::init(COLLISION_MESHES, true).expect("initialize embedded RocketSim Soccar collision meshes");

    if self_test {
        std::process::exit(self_test::run(model_arg.as_deref()));
    }

    // Grab environment variables injected by RLBot (Port, ID).
    let AgentEnvironment {
        server_addr,
        agent_id,
    } = AgentEnvironment::from_env();
    
    let agent_id = agent_id.unwrap_or_else(|| "buta_ppo/buta-ppo-bot".into());
    let connection = RLBotConnection::new(&server_addr).expect("connect to RLBot");

    println!("Starting agent: {}...", agent_id);
    run_bot_agents::<ButaPPOBot>(agent_id, false, false, connection).expect("run buta_ppo RLBot agent");
}
