import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def deploy():
    # LibTorch location, from the same environment variable the CMake build uses. Cargo's build script reads it too.
    libtorch = os.environ.get("LIBTORCH")
    if not libtorch or not (Path(libtorch) / "lib").is_dir():
        sys.exit("Set the LIBTORCH environment variable to your LibTorch directory (the one containing lib/ and share/).")
    torch_lib_dir = Path(libtorch) / "lib"

    print("Building buta_ppo_bot via Cargo...")
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "buta_ppo_bot"], 
        cwd="rust_bridge", 
        check=True
    )

    bot_dir = Path("bot")
    bot_lib_dir = Path("bot/libs")
    bot_lib_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying executable to {bot_dir.absolute()}...")
    shutil.copy("rust_bridge/target/release/buta_ppo_bot", bot_dir)

    # Cargo keeps a build directory per build script configuration, so older copies of the library can sit next to
    # the current one. The most recently built is the one the build above produced or kept.
    print("Locating custom C++ shared library...")
    libs = list(Path("rust_bridge/target/release/build").rglob("libbuta_ppo_bot.so"))
    if libs:
        newest = max(libs, key=lambda p: p.stat().st_mtime)
        shutil.copy(newest, bot_lib_dir)
        built = datetime.fromtimestamp(newest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  -> Copied {newest} (built {built})")
    else:
        print("  -> Warning: libbuta_ppo_bot.so not found in rust_bridge/target/release/build")

    print("Copying LibTorch dependencies...")
    torch_libs = ["libc10.so", "libc10_cuda.so", "libtorch.so", "libtorch_cpu.so"]
    for lib in torch_libs:
        lib_path = torch_lib_dir / lib
        if lib_path.exists():
            shutil.copy(lib_path, bot_lib_dir)
            print(f"  -> Copied {lib}")
        else:
            print(f"  -> Warning: {lib} not found in {torch_lib_dir}")

    print("\nDeployment complete. Remaining steps:")
    print("  1. Copy checkpoints/<bot_name>_<steps>/model.pt into bot/.")
    print("  2. Copy assets/collision_meshes/ into bot/.")
    print("  3. Change the TEMPLATE values in bot/bot.toml and bot/loadout.toml for your bot.")

if __name__ == "__main__":
    deploy()