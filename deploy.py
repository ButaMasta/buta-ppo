import shutil
import subprocess
from pathlib import Path

def deploy():
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

    print("Locating custom C++ shared library...")
    for path in Path("rust_bridge/target/release/build").rglob("libbuta_ppo_bot.so"):
        shutil.copy(path, bot_lib_dir)
        print(f"  -> Copied {path.name}")
        break
    else:
        print("  -> Warning: libbuta_ppo_bot.so not found in rust_bridge/target/release/build")

    print("Copying LibTorch dependencies...")
    torch_lib_dir = Path("/home/buta/Dev/libtorch/lib")
    
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