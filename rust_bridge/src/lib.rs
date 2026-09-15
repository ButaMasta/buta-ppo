use std::ffi::CStr;
use std::os::raw::c_char;
use rocketsim::{Arena, GameMode, Team, CarBodyConfig};

//// Mesh initialization. ////
#[no_mangle]
pub extern "C" fn rs_init_from_default(silent: bool) -> bool {
    rocketsim::init_from_default(silent).is_ok()
}

#[no_mangle]
pub extern "C" fn rs_init(collision_meshes_folder: *const c_char, silent: bool) -> bool {
    if collision_meshes_folder.is_null() {
        return false;
    }
    
    // Parse the C-string pointer into a Rust string slice.
    let c_str = unsafe { CStr::from_ptr(collision_meshes_folder) };
    let path_str = match c_str.to_str() {
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

    let arena = Arena::new(game_mode);
    
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
/// Steps the arena for 1 tick, returning the events produced during that tick.
#[no_mangle]
pub extern "C" fn rs_arena_step(arena_ptr: *mut Arena) {
    if arena_ptr.is_null() {
        return;
    }

    // Convert raw C pointer into a Rust mutable reference.
    let arena = unsafe { &mut *arena_ptr };
    
    // Advance the physics engine by one tick.
    arena.step_tick();
}

/// Creates and adds a car to the arena, returning the index of the car in the cars vector.
#[no_mangle]
pub extern "C" fn rs_arena_add_car(arena_ptr: *mut Arena, team_idx: i32) -> u32 {
    if arena_ptr.is_null() {
        return 0;
    }

    let arena = unsafe { &mut *arena_ptr };
    
    // Map integer to Team (0 = Blue, 1 = Orange).
    let team = if team_idx == 0 { Team::Blue } else { Team::Orange };
    
    let config = CarBodyConfig::OCTANE; 
    
    // add_car returns the index of the newly added car (usize).
    // Cast to u32 for standard C-ABI compatibility.
    arena.add_car(team, config) as u32
}

#[no_mangle]
pub extern "C" fn rs_arena_get_basic_obs(
    arena_ptr: *const Arena, 
    car_idx: u32, 
    out_obs: *mut f32
) {
    if arena_ptr.is_null() || out_obs.is_null() {
        return;
    }

    let arena = unsafe { &*arena_ptr };
    
    // Retrieve states from RocketSim
    let ball_state = arena.get_ball_state();
    let car_state = arena.get_car_state(car_idx as usize);

    // Safely wrap the raw C pointer into a Rust mutable slice of exactly 6 floats
    let obs = unsafe { std::slice::from_raw_parts_mut(out_obs, 6) };

    // Pack the Ball position
    obs[0] = ball_state.phys.pos.x;
    obs[1] = ball_state.phys.pos.y;
    obs[2] = ball_state.phys.pos.z;

    // Pack the Car position
    obs[3] = car_state.phys.pos.x;
    obs[4] = car_state.phys.pos.y;
    obs[5] = car_state.phys.pos.z;
}