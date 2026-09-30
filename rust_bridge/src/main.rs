// buta-ppo/rust_bridge/src/main.rs
mod bot_ffi;

use std::ffi::CString;
use std::os::raw::c_void;
use std::sync::Arc;

use rust_bridge::{CArenaState, CBallState, CCarControls, CCarState, CPhysState};
use crate::bot_ffi::{bot_api_free, bot_api_get_action, bot_api_init};

use rlbot_rocketsim::rlbot::agents::{run_bot_agents, BotAgent};
use rlbot_rocketsim::rlbot::flat::{
    ControllableInfo, FieldInfo, GamePacket, MatchConfiguration, MatchPhase, PlayerInput,
};
use rlbot_rocketsim::rlbot::util::{AgentEnvironment, PacketQueue};
use rlbot_rocketsim::rlbot::RLBotConnection;
use rlbot_rocketsim::{GameStateEnricher, MatchContext};

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
}

impl BotAgent for ButaPPOBot {
    fn new(
        _team: u32,
        controllable_info: ControllableInfo,
        match_config: Arc<MatchConfiguration>,
        field_info: Arc<FieldInfo>,
        _packet_queue: &mut PacketQueue,
    ) -> Self {
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

        let path_str = "model.pt";
        let path_c = CString::new(path_str).expect("Invalid CString");
        let ctx_ptr = unsafe { bot_api_init(path_c.as_ptr()) };
        
        if ctx_ptr.is_null() {
            panic!("Failed to load C++ LibTorch checkpoint at: {}", path_str);
        }

        Self {
            player_index: controllable_info.index as usize,
            enricher: GameStateEnricher::from_match_context(context),
            ctx_ptr,
            held_controls: CCarControls::default(),
            next_decision_frame: None,
            tick_skip: 8,
            big_pad_indices,
            small_pad_indices,
        }
    }

    fn tick(&mut self, packet: &GamePacket, packet_queue: &mut PacketQueue) {
        let frame = packet.match_info.frame_num;
        
        // Feed the RLBot packet into the RocketSim enricher.
        let enriched = self.enricher.update(packet).is_ok();

        // Only infer new actions during Kickoff or Active gameplay.
        if enriched
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
                    is_on_ground: rs_car_state.is_on_ground,
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
            c_arena.boost_pads.big[i] = packet.boost_pads[idx].timer;
        }
        for (i, &idx) in self.small_pad_indices.iter().enumerate() {
            c_arena.boost_pads.small[i] = packet.boost_pads[idx].timer;
        }

        // Execute LibTorch Forward Pass.
        self.held_controls = unsafe { bot_api_get_action(self.ctx_ptr, &c_arena, my_car_idx) };
        self.next_decision_frame = Some(frame + self.tick_skip);
    }
}

impl Drop for ButaPPOBot {
    fn drop(&mut self) {
        unsafe { bot_api_free(self.ctx_ptr) };
    }
}

fn main() {
    rocketsim::init("collision_meshes", true).expect("initialize embedded RocketSim Soccar collision meshes");
    ball_sim::init("collision_meshes", true).expect("initialize embedded BallSim Soccar collision meshes");

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