// buta-ppo/rust_bridge/bot/src/self_test.rs
//! `buta_ppo_bot --self-test [model]`: checks the bot runs on this machine without RLBot or the game. It loads the
//! model through the C++ library, as a match would, then plays a few decisions from a seeded RocketSim kickoff in each
//! team size the model supports and prints every car's controls.
//!
//! The model is the one given, else model.pt if it exists, else a small 1v1 test model with seeded random weights that
//! the C++ library writes for its current obs and action code, so the test runs anywhere, e.g. in CI.
//!
//! 1v1 output of a 1v1 model is deterministic, so builds on different OSes or compilers can be compared line by line
//! (near-tie argmax choices may still differ). Anything with more car slots than cars isn't: the obs builder shuffles
//! the slots with a random seed.

use std::ffi::CString;

use rust_bridge::{
    rs_arena_add_car, rs_arena_create, rs_arena_free, rs_arena_get_arena_state, rs_arena_reset_to_random_kickoff,
    rs_arena_set_car_controls, rs_arena_step, CArenaState, CCarControls,
};

use crate::bot_ffi::{
    bot_api_free, bot_api_get_action, bot_api_init, bot_api_max_players_per_team, bot_api_tick_skip,
    bot_api_write_test_model,
};
use crate::COLLISION_MESHES;

/// Decisions played per team size.
const DECISIONS: u32 = 8;

fn format_controls(c: &CCarControls) -> String {
    let flag = |on: bool, name: char| if on { name } else { '-' };
    // Adding 0 turns the -0 of mirrored zero inputs into +0.
    let axis = |value: f32| value + 0.0;
    format!(
        "{:+.2} {:+.2} {:+.2} {:+.2} {:+.2} {}{}{}",
        axis(c.throttle),
        axis(c.steer),
        axis(c.pitch),
        axis(c.yaw),
        axis(c.roll),
        flag(c.jump, 'j'),
        flag(c.boost, 'b'),
        flag(c.handbrake, 'h')
    )
}

/// The test model's team size and tick skip. 1v1 only: with more car slots than cars the obs builder shuffles them with a
/// random seed, and std::shuffle differs between C++ standard libraries anyway, so larger sizes aren't reproducible.
const TEST_MODEL_MAX_PLAYERS_PER_TEAM: u32 = 1;
const TEST_MODEL_TICKS_PER_STEP: u32 = 8;

/// Runs the self-test with the given model (see the module docs for the default) and returns the process exit code.
pub fn run(model: Option<&str>) -> i32 {
    let meshes_c = CString::new(COLLISION_MESHES).expect("mesh path without NUL bytes");
    // The test model is written to a temporary file, removed afterwards.
    let mut temp_model = None;
    let model = match model {
        Some(model) => model.to_owned(),
        None if std::path::Path::new("model.pt").is_file() => "model.pt".to_owned(),
        None => {
            let path = std::env::temp_dir().join(format!("buta_ppo_bot_self_test_{}.pt", std::process::id()));
            let path = path.to_string_lossy().into_owned();
            let path_c = CString::new(path.as_str()).expect("temp path without NUL bytes");
            let written = unsafe {
                bot_api_write_test_model(path_c.as_ptr(), meshes_c.as_ptr(), TEST_MODEL_MAX_PLAYERS_PER_TEAM,
                                         TEST_MODEL_TICKS_PER_STEP)
            };
            temp_model = Some(path.clone());
            if !written {
                eprintln!("[buta_ppo_bot] Self-test failed: couldn't write a test model to {path}.");
                let _ = std::fs::remove_file(&path);
                return 1;
            }
            println!("Self-test: no model.pt, so using a test model with seeded random weights.");
            path
        }
    };
    let code = run_model(&model, &meshes_c);
    if let Some(path) = temp_model {
        let _ = std::fs::remove_file(path);
    }
    code
}

fn run_model(model: &str, meshes_c: &CString) -> i32 {
    let model_c = CString::new(model).expect("model path without NUL bytes");
    let ctx = unsafe { bot_api_init(model_c.as_ptr(), meshes_c.as_ptr()) };
    if ctx.is_null() {
        eprintln!("[buta_ppo_bot] Self-test failed: couldn't load {model}.");
        return 1;
    }
    let max_per_team = unsafe { bot_api_max_players_per_team(ctx) };
    let tick_skip = unsafe { bot_api_tick_skip(ctx) };
    println!("Self-test: {DECISIONS} decisions per team size, {tick_skip} ticks each.");
    println!("Controls: throttle steer pitch yaw roll, then jump/boost/handbrake.");

    for team_size in 1..=max_per_team {
        let arena = rs_arena_create(0);
        for team in 0..2 {
            for _ in 0..team_size {
                rs_arena_add_car(arena, team);
            }
        }
        rs_arena_reset_to_random_kickoff(arena, u64::from(team_size), true);

        let mut state = CArenaState::default();
        for decision in 0..DECISIONS {
            rs_arena_get_arena_state(arena, &mut state);
            let mut line = format!("{team_size}v{team_size} #{decision}:");
            for car in 0..state.num_cars {
                // Cars decide in order on the same state, and the first decision is a kickoff (slot reshuffle).
                let controls = unsafe { bot_api_get_action(ctx, &state, car, decision == 0) };
                rs_arena_set_car_controls(arena, car, controls);
                line += &format!(" | {}", format_controls(&controls));
            }
            println!("{line}");
            for _ in 0..tick_skip {
                rs_arena_step(arena);
            }
        }
        rs_arena_free(arena);
    }

    unsafe { bot_api_free(ctx) };
    println!("Self-test passed.");
    0
}
