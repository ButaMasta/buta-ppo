// buta-ppo/rust_bridge/src/lib.rs
use std::{ffi::CStr};
use std::os::raw::c_char;
use glam::{Mat3A, Vec3A};
use rocketsim::{
    Arena, ArenaEvent, BallState, Car, CarBodyConfig, CarControls, CarInfo, CarState, GameMode, PhysState, Team
};
use rocketsim_vis::ArenaVisExt;

//// C++ State Interface Structs. ////

/// Core states.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CPhysState {
    pub pos: [f32; 3],
    pub rot_mat: [[f32; 3]; 3],
    pub vel: [f32; 3],
    pub ang_vel: [f32; 3],
}

impl From<&PhysState> for CPhysState {
    fn from(phys: &PhysState) -> Self {
        Self {
            pos: phys.pos.into(),
            rot_mat: [
                phys.rot_mat.x_axis.into(),
                phys.rot_mat.y_axis.into(),
                phys.rot_mat.z_axis.into(),
            ],
            vel: phys.vel.into(),
            ang_vel: phys.ang_vel.into(),
        }
    }
}

impl From<CPhysState> for PhysState {
    fn from(c: CPhysState) -> Self {
        Self {
            pos: Vec3A::from_array(c.pos),
            rot_mat: Mat3A {
                x_axis: Vec3A::from_array(c.rot_mat[0]),
                y_axis: Vec3A::from_array(c.rot_mat[1]),
                z_axis: Vec3A::from_array(c.rot_mat[2]),
            },
            vel: Vec3A::from_array(c.vel),
            ang_vel: Vec3A::from_array(c.ang_vel),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CBallState {
    pub phys: CPhysState,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CCarState {
    pub phys: CPhysState,
    pub team: i32,
    pub boost: f32,
    pub handbrake_val: f32,
    pub air_time_since_jump: f32,
    pub demo_respawn_timer: f32,
    pub is_on_ground: bool,
    pub has_jumped: bool,
    pub has_double_jumped: bool,
    pub is_jumping: bool,
    pub has_flip_or_jump: bool,
    pub has_flipped: bool,
    pub is_auto_flipping: bool,
    pub is_flipping: bool,
    pub is_demoed: bool,
    pub is_supersonic: bool,
}

/// Events and Global State.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CArenaEvents {
    pub is_ball_scored: bool,
    pub car_hit_ball: [bool; 8],
    pub car_got_boost: [bool; 8],
    pub ball_hit_extra_vel: [f32; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CArenaState {
    pub ball: CBallState,
    pub cars: [CCarState; 8],
    pub pads: [f32; 34],
    pub num_cars: u32,
    pub tick_count: u64,
    pub events: CArenaEvents,
}

impl Default for CArenaState {
    fn default() -> Self {
        Self {
            ball: Default::default(),
            cars: [Default::default(); 8],
            pads: [0.0; 34],
            num_cars: 0,
            tick_count: 0,
            events: Default::default(),
        }
    }
}

/// Input Struct.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CCarControls {
    pub throttle: f32,
    pub steer: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
    pub jump: bool,
    pub boost: bool,
    pub handbrake: bool,
}

// Convert RocketSim's controls from C++ to rust.
impl From<CCarControls> for CarControls {
    fn from(c: CCarControls) -> Self {
        Self {
            throttle: c.throttle,
            steer: c.steer,
            pitch: c.pitch,
            yaw: c.yaw,
            roll: c.roll,
            jump: c.jump,
            boost: c.boost,
            handbrake: c.handbrake,
        }
    }
}


//// Mesh initialization. ////
#[no_mangle]
pub extern "C" fn rs_init(collision_meshes_folder: *const c_char, silent: bool) -> bool {
    if collision_meshes_folder.is_null() {
        return false;
    }
    
    // Parse the C-string pointer into a Rust string slice.
    let c_str: &CStr = unsafe { CStr::from_ptr(collision_meshes_folder) };
    let path_str: &str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    rocketsim::init(path_str, silent).is_ok()
}

#[no_mangle]
pub extern "C" fn rs_is_initialized() -> bool {
    rocketsim::is_initialized()
}


//// Arena Creation. ////
#[no_mangle]
pub extern "C" fn rs_arena_create_vis(game_mode_idx: i32) -> *mut Arena {
    // Map the integer to the RocketSim enum. Defaulting to Soccar (0).
    let game_mode = match game_mode_idx {
        0 => GameMode::Soccar,
        1 => GameMode::Hoops,
        2 => GameMode::Heatseeker,
        3 => GameMode::Snowday,
        4 => GameMode::Dropshot,
        _ => GameMode::Soccar, 
    };

    let mut arena: Arena = Arena::new(game_mode);
    arena.set_vis_enabled(true);
    
    // Move the Arena to the heap and hand the pointer to C++.
    Box::into_raw(Box::new(arena))
}

#[no_mangle]
pub extern "C" fn rs_arena_create(game_mode_idx: i32) -> *mut Arena {
    // Map the integer to the RocketSim enum. Defaulting to Soccar (0).
    let game_mode = match game_mode_idx {
        0 => GameMode::Soccar,
        1 => GameMode::Hoops,
        2 => GameMode::Heatseeker,
        3 => GameMode::Snowday,
        4 => GameMode::Dropshot,
        _ => GameMode::Soccar, 
    };

    let arena: Arena = Arena::new(game_mode);
    
    // Move the Arena to the heap and hand the pointer to C++.
    Box::into_raw(Box::new(arena))
}

#[no_mangle]
pub extern "C" fn rs_arena_free(arena_ptr: *mut Arena) {
    if !arena_ptr.is_null() {
        // Reconstruct the Box from the raw pointer so Rust can drop it.
        unsafe {
            let _ = Box::from_raw(arena_ptr);
        }
    }
}


//// Arena Util. ////
/// Creates and adds a car to the arena, returning the index of the car in the cars vector.
#[no_mangle]
pub extern "C" fn rs_arena_add_car(arena_ptr: *mut Arena, team_idx: i32) -> u32 {
    if arena_ptr.is_null() {
        return 0;
    }

    let arena: &mut Arena = unsafe { &mut *arena_ptr };
    
    // Map integer to Team (0 = Blue, 1 = Orange).
    let team: Team = if team_idx == 0 { Team::Blue } else { Team::Orange };
    
    let config: CarBodyConfig = CarBodyConfig::OCTANE; 
    
    // add_car returns the index of the newly added car (usize).
    // Cast to u32 for standard C-ABI compatibility.
    arena.add_car(team, config) as u32
}

/// Steps the arena for 1 tick, returning the events produced during that tick.
#[no_mangle]
pub extern "C" fn rs_arena_step(arena_ptr: *mut Arena) {
    if arena_ptr.is_null() {
        return;
    }

    // Convert raw C pointer into a Rust mutable reference.
    let arena: &mut Arena = unsafe { &mut *arena_ptr };
    
    // Advance the physics engine by one tick.
    arena.step_tick();
}

/// Core function to grab the entire arena state in one call.
#[no_mangle]
pub extern "C" fn rs_arena_get_arena_state(
    arena_ptr: *const Arena,
    out_state: *mut CArenaState,
) {
    if arena_ptr.is_null() || out_state.is_null() {
        return;
    }

    let arena: &Arena = unsafe { &*arena_ptr };
    let state: &mut CArenaState = unsafe { &mut *out_state };

    // For now just clear the state to be safe.
    *state = CArenaState::default();

    state.tick_count = arena.tick_count();

    // Pack ball.
    let ball: &BallState = arena.get_ball_state();
    state.ball.phys = CPhysState::from(&ball.phys);

    // Pack cars.
    let cars: &Vec<Car> = arena.cars();
    state.num_cars = cars.len().min(8) as u32;

    for (i, car) in cars.iter().enumerate().take(8) {
        let car_state: &CarState = car.get_state();
        let car_info: &CarInfo = arena.get_car_info(i);

        state.cars[i].phys = CPhysState::from(&car_state.phys);

        state.cars[i].team = if car_info.team == Team::Blue { 0 } else { 1 };
        state.cars[i].boost = car_state.boost;
        state.cars[i].handbrake_val = car_state.handbrake_val;
        state.cars[i].air_time_since_jump = car_state.air_time_since_jump;
        state.cars[i].demo_respawn_timer = car_state.demo_respawn_timer;
        state.cars[i].is_on_ground = car_state.is_on_ground;
        state.cars[i].has_jumped = car_state.has_jumped;
        state.cars[i].has_double_jumped = car_state.has_double_jumped;
        state.cars[i].is_jumping = car_state.is_jumping;
        state.cars[i].has_flip_or_jump = car_state.has_flip_or_jump();
        state.cars[i].has_flipped = car_state.has_flipped;
        state.cars[i].is_auto_flipping = car_state.is_auto_flipping;
        state.cars[i].is_flipping = car_state.is_flipping;
        state.cars[i].is_demoed = car_state.is_demoed;
        state.cars[i].is_supersonic = car_state.is_supersonic;
    }

    // Pack events.
    state.events.is_ball_scored = arena.is_ball_scored();

    for event in arena.get_last_step_events() {
        match event {
            ArenaEvent::CarHitBall(hit) => {
                let idx: usize = hit.car_idx as usize;
                if idx < 8 {
                    state.events.car_hit_ball[idx] = true;
                    state.events.ball_hit_extra_vel[idx] = hit.extra_hit_vel.length();
                }
            }
            ArenaEvent::CarPickupBoost(pickup) => {
                let idx: usize = pickup.car_idx as usize;
                if idx < 8 {
                    state.events.car_got_boost[idx] = true;
                }
            }
            _ => {}
        }
    }
}

/// Sets the car controls.
#[no_mangle]
pub extern "C" fn rs_arena_set_car_controls(
    arena_ptr: *mut Arena,
    car_idx: u32,
    controls: CCarControls,
) {
    if arena_ptr.is_null() { return; }
    let arena: &mut Arena = unsafe { &mut *arena_ptr };

    arena.set_car_controls(car_idx as usize, CarControls::from(controls));
}

/// Set ball phys state.
#[no_mangle]
pub extern "C" fn rs_arena_set_ball_state(arena_ptr: *mut Arena, ball_state: CBallState) {
    if arena_ptr.is_null() {
        return;
    }
    let arena: &mut Arena = unsafe { &mut *arena_ptr };

    let mut curr_ball: BallState = *arena.get_ball_state();
    curr_ball.phys = PhysState::from(ball_state.phys);
    arena.set_ball_state(curr_ball);
}

/// Set a car's state, phys and flags/amounts.
#[no_mangle]
pub extern "C" fn rs_arena_set_car_state(
    arena_ptr: *mut Arena,
    car_idx: u32,
    car_state: CCarState,
) {
    if arena_ptr.is_null() {
        return;
    }
    let arena: &mut Arena = unsafe { &mut *arena_ptr };
    let idx: usize = car_idx as usize;

    if idx >= arena.num_cars() {
        return;
    }

    let mut curr_car: CarState = *arena.get_car_state(idx);
    curr_car.phys                = PhysState::from(car_state.phys);
    curr_car.boost               = car_state.boost;
    curr_car.handbrake_val       = car_state.handbrake_val;
    curr_car.air_time_since_jump = car_state.air_time_since_jump;
    curr_car.demo_respawn_timer  = car_state.demo_respawn_timer;
    curr_car.is_on_ground        = car_state.is_on_ground;
    curr_car.has_jumped          = car_state.has_jumped;
    curr_car.has_double_jumped   = car_state.has_double_jumped;
    curr_car.is_jumping          = car_state.is_jumping;
    curr_car.has_flipped         = car_state.has_flipped;
    curr_car.is_auto_flipping    = car_state.is_auto_flipping;
    curr_car.is_flipping         = car_state.is_flipping;
    curr_car.is_demoed           = car_state.is_demoed;
    curr_car.is_supersonic       = car_state.is_supersonic;

    arena.set_car_state(idx, curr_car);
}

/// Reset arena to random kickoff state.
#[no_mangle]
pub extern "C" fn rs_arena_reset_to_random_kickoff(
    arena_ptr: *mut Arena,
    seed: u64,
    use_seed: bool,
) {
    if arena_ptr.is_null() {
        return;
    }
    let arena: &mut Arena = unsafe { &mut *arena_ptr };
    let rng_seed: Option<u64> = if use_seed { Some(seed) } else { None };
    arena.reset_to_random_kickoff(rng_seed);
}