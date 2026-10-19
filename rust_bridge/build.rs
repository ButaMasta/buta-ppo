// buta-ppo/rust_bridge/build.rs
fn main() {
    // LibTorch location, from the same environment variable the CMake build uses.
    let torch_dir = std::env::var("LIBTORCH")
        .expect("Set the LIBTORCH environment variable to your LibTorch directory (the one containing lib/ and share/).");
    println!("cargo:rerun-if-env-changed=LIBTORCH");

    // Compile the C++ bot library using CMake.
    let dst = cmake::Config::new("..")
        .define("BUILD_RUST_BRIDGE", "OFF")
        .define("CMAKE_CXX_STANDARD", "20")
        .define("LIBTORCH_DIR", &torch_dir)
        .build_target("buta_ppo_bot")
        .build();

    // Link the newly built C++ library.
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=dylib=buta_ppo_bot");

    // Link LibTorch dependencies so Rust can resolve the C++ symbols.
    println!("cargo:rustc-link-search=native={}/lib", torch_dir);
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    // Re-run this script when any C++ source, header, or CMake file changes. Cargo scans directories recursively.
    // Headers matter as much as sources: the obs coefficients and the FFI struct layouts live in them, and a stale
    // library would build a different obs than training did without any error. CMake then rebuilds only what changed.
    println!("cargo:rerun-if-changed=../src");
    println!("cargo:rerun-if-changed=../include");
    println!("cargo:rerun-if-changed=../cmake");
    println!("cargo:rerun-if-changed=../CMakeLists.txt");

    // Instruct the dynamic linker to check the executable's directory for .so files.
    println!("cargo:rustc-link-arg=-Wl,-rpath=$ORIGIN/libs");
}
