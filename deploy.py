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
    bot_dir.mkdir(exist_ok=True)

    print(f"Copying executable to {bot_dir.absolute()}...")
    shutil.copy("rust_bridge/target/release/buta_ppo_bot", bot_dir)

    print("Locating custom C++ shared library...")
    for path in Path("rust_bridge/target/release/build").rglob("libbuta_ppo_bot.so"):
        shutil.copy(path, bot_dir)
        print(f"  -> Copied {path.name}")
        break

    print("Copying LibTorch dependencies...")
    torch_lib_dir = Path("/home/buta/Dev/libtorch/lib")
    
    # Add libtorch_cuda.so and libc10_cuda.so to this list if executing on the GPU.
    torch_libs = ["libc10.so", "libc10_cuda.so", "libtorch.so", "libtorch_cpu.so"]
    for lib in torch_libs:
        lib_path = torch_lib_dir / lib
        if lib_path.exists():
            shutil.copy(lib_path, bot_dir)
            print(f"  -> Copied {lib}")
        else:
            print(f"  -> Warning: {lib} not found in {torch_lib_dir}")

    print("\nDeployment complete. You can now manually place `model.pt`, `bot.toml`, and `collision_meshes/` into the bot/ directory.")

if __name__ == "__main__":
    deploy()