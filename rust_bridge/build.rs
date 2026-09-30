// buta-ppo/rust_bridge/build.rs
fn main() {
    // Compile the C++ bot library using CMake.
    let dst = cmake::Config::new("..")
        .define("BUILD_RUST_BRIDGE", "OFF")
        .define("CMAKE_CXX_STANDARD", "20")
        .define("Torch_DIR", "/home/buta/Dev/libtorch/share/cmake/Torch")
        .build_target("buta_ppo_bot")
        .build();

    // Link the newly built C++ library.
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=dylib=buta_ppo_bot");

    // Link LibTorch dependencies so Rust can resolve the C++ symbols.
    let torch_dir = "/home/buta/Dev/libtorch";
    println!("cargo:rustc-link-search=native={}/lib", torch_dir);
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    // Tell Cargo to re-run this script if any C++ source files change.
    println!("cargo:rerun-if-changed=../src/bot_api.cpp");
    println!("cargo:rerun-if-changed=../src/obs/advanced_obs.cpp");
    println!("cargo:rerun-if-changed=../src/action/default_action.cpp");
    println!("cargo:rerun-if-changed=../src/rl/actor_critic.cpp");

    // Instruct the dynamic linker to check the executable's directory for .so files.
    println!("cargo:rustc-link-arg=-Wl,-rpath=$ORIGIN");
}