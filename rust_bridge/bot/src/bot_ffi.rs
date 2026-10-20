// buta-ppo/rust_bridge/bot/src/bot_ffi.rs
use std::ffi::c_void;
use std::os::raw::c_char;

use rust_bridge::{CArenaState, CCarControls};

unsafe extern "C" {
    /// Loads the model and builds the obs/action builders from the config stored in it.
    /// Returns a pointer to the BotContext, or null if the model couldn't be loaded.
    pub unsafe fn bot_api_init(checkpoint_path: *const c_char) -> *mut c_void;

    /// Infers the model for the given state and car then returns the controls.
    /// `new_kickoff` reshuffles the obs car slots (once per kickoff).
    pub unsafe fn bot_api_get_action(
        ctx_ptr: *mut c_void,
        state: *const CArenaState,
        car_idx: u32,
        new_kickoff: bool,
    ) -> CCarControls;

    /// Returns the max players per team the model was trained with.
    pub unsafe fn bot_api_max_players_per_team(ctx_ptr: *mut c_void) -> u32;

    /// Returns the ticks per step the model was trained with, the frames between decisions.
    pub unsafe fn bot_api_tick_skip(ctx_ptr: *mut c_void) -> u32;

    /// Frees the C++ context memory.
    pub unsafe fn bot_api_free(ctx_ptr: *mut c_void);
}