// buta-ppo/rust_bridge/src/bot_ffi.rs
use std::ffi::c_void;
use std::os::raw::c_char;

use rust_bridge::{CArenaState, CCarControls};

unsafe extern "C" {
    /// Initializes the LibTorch model and obs/action builders.
    /// Returns a pointer to the BotContext.
    pub unsafe fn bot_api_init(checkpoint_path: *const c_char) -> *mut c_void;

    /// Infers the model for the given state and car then returns the controls.
    pub unsafe fn bot_api_get_action(
        ctx_ptr: *mut c_void,
        state: *const CArenaState,
        car_idx: u32,
    ) -> CCarControls;

    /// Frees the C++ context memory.
    pub unsafe fn bot_api_free(ctx_ptr: *mut c_void);
}