// buta-ppo/rust_bridge/bot/build.rs
fn main() {
    // Build scripts run on the host, so cfg!(windows) is the host and CARGO_CFG_TARGET_OS the target.
    let target_windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");

    // LibTorch location, from the same environment variable the CMake build uses. A Windows bot built on another OS
    // needs the Windows LibTorch, from LIBTORCH_WINDOWS (which a Windows host may set instead of LIBTORCH too).
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_WINDOWS");
    let torch_dir = libtorch_dir(target_windows).unwrap_or_else(|| {
        if target_windows && !cfg!(windows) {
            panic!("Set the LIBTORCH_WINDOWS environment variable to the Windows CPU LibTorch directory (the one \
                    containing lib/ and share/).");
        }
        panic!("Set the LIBTORCH environment variable to your LibTorch directory (the one containing lib/ and share/).");
    });

    let bridge_lib = rust_bridge_lib(target_windows);

    // Compile the C++ bot library using CMake.
    let mut config = cmake::Config::new("../..");
    config
        .define("BUILD_RUST_BRIDGE", "OFF")
        .define("RUST_BRIDGE_LIB", &bridge_lib)
        .define("CMAKE_CXX_STANDARD", "20")
        .define("LIBTORCH_DIR", &torch_dir)
        // CMakeLists.txt prefers the LIBTORCH environment variable to LIBTORCH_DIR, so it gets the target's too.
        .env("LIBTORCH", &torch_dir)
        .build_target("buta_ppo_bot");
    if target_windows {
        // The library uses the DLL C runtime like LibTorch (see CMakeLists.txt), and this crate would otherwise pass
        // the flags of this binary's static one.
        config.static_crt(false);
    }
    let dst = config.build();

    // Link the newly built C++ library. Multi-config generators (Visual Studio) put it in a Release/ subdirectory.
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}/build/Release", dst.display());
    println!("cargo:rustc-link-lib=dylib=buta_ppo_bot");

    // Re-run this script when any C++ source, header, or CMake file changes. Cargo scans directories recursively.
    // Headers matter as much as sources: the obs coefficients and the FFI struct layouts live in them, and a stale
    // library would build a different obs than training did without any error. CMake then rebuilds only what changed.
    println!("cargo:rerun-if-changed=../../src");
    println!("cargo:rerun-if-changed=../../include");
    println!("cargo:rerun-if-changed=../../cmake");
    println!("cargo:rerun-if-changed=../../CMakeLists.txt");
    println!("cargo:rerun-if-changed={}", bridge_lib.display());

    if target_windows {
        // Windows has no rpath. The DLL is loaded only on its first call, after main has loaded it from libs/ (see
        // load_bot_library in main.rs), so the binary starts without it next to it.
        println!("cargo:rustc-link-arg=/DELAYLOAD:buta_ppo_bot.dll");
        println!("cargo:rustc-link-lib=delayimp");
    } else {
        // The binary only calls the C++ library, but the linker checks the library's own LibTorch dependencies too.
        println!("cargo:rustc-link-arg=-Wl,-rpath-link={}/lib", torch_dir);
        // Instruct the dynamic linker to check libs/ next to the executable for the C++ library.
        println!("cargo:rustc-link-arg=-Wl,-rpath=$ORIGIN/libs");
        // Export nothing. This binary has its own copy of the Rust bridge, and the linker would export the bridge
        // functions the C++ library calls, which would then bind to this copy instead of the library's own, leaving a
        // mix of the two (Windows never resolves a DLL's imports against the executable).
        let script = std::path::Path::new(&std::env::var("OUT_DIR").expect("Cargo sets OUT_DIR")).join("exports.ver");
        std::fs::write(&script, "{ local: *; };\n").expect("write the version script");
        println!("cargo:rustc-link-arg=-Wl,--version-script={}", script.display());
    }
}

/// The Rust bridge library the C++ library links: the `rust_bridge` package's cdylib (on Windows its import library) in
/// this build's target directory. It has to be built first, in a separate cargo run, because Cargo builds it alongside
/// this script rather than before it.
fn rust_bridge_lib(target_windows: bool) -> std::path::PathBuf {
    // OUT_DIR is <target dir>[/<triple>]/<profile>/build/<package>-<hash>/out.
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let profile_dir = out_dir.ancestors().nth(3).expect("OUT_DIR is inside the target directory");
    let lib = profile_dir.join(if target_windows { "rust_bridge.dll.lib" } else { "librust_bridge.so" });
    // The same --target as this build, if it had one (the profile directory is then inside a directory named for it).
    let target = std::env::var("TARGET").expect("Cargo sets TARGET");
    let explicit_target = profile_dir.parent().and_then(|p| p.file_name()).is_some_and(|name| *name == *target);
    let target_arg = if explicit_target { format!(" --target {target}") } else { String::new() };
    let cargo_build = if target_windows && !cfg!(windows) { "cargo xwin build" } else { "cargo build" };
    let build_command = format!("{cargo_build} --release -p rust_bridge{target_arg} (deploy.py does this)");

    let modified = |path: &std::path::Path| std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let Some(lib_time) = modified(&lib) else {
        panic!("{} not found. Build the Rust bridge first: {build_command}.", lib.display());
    };
    // A bridge older than its source would link the C++ library against stale code without any error.
    let sources = std::fs::read_dir("../src").expect("rust_bridge/src exists").flatten().map(|e| e.path());
    for source in sources.chain([std::path::PathBuf::from("../Cargo.toml")]) {
        if modified(&source).is_some_and(|t| t > lib_time) {
            panic!("{} is older than {}. Build the Rust bridge first: {build_command}.", lib.display(),
                   source.display());
        }
    }
    lib
}

/// The LibTorch directory for the target: LIBTORCH_WINDOWS for a Windows target, falling back to LIBTORCH when the
/// target is the host OS.
fn libtorch_dir(target_windows: bool) -> Option<String> {
    let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    let same_os = target_windows == cfg!(windows);
    if target_windows {
        var("LIBTORCH_WINDOWS").or_else(|| if same_os { var("LIBTORCH") } else { None })
    } else if same_os {
        var("LIBTORCH")
    } else {
        None
    }
}
