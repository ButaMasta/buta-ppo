import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

def find_model(checkpoint, bot_name):
    """The model file to deploy: the given checkpoint, or else the latest checkpoint in checkpoints/."""
    if checkpoint:
        path = Path(checkpoint)
        model = path / "model.pt" if path.is_dir() else path
        if not model.is_file():
            sys.exit(f"No model at {model}. Pass a checkpoint directory or a .pt model file.")
        return model

    # Checkpoint directories are <bot_name>_<steps>. The PFSP pool and interrupted .tmp saves don't end in a number.
    latest = {}
    for path in Path("checkpoints").glob("*_*"):
        name, _, steps = path.name.rpartition("_")
        if steps.isdecimal() and (path / "model.pt").is_file() and int(steps) > latest.get(name, (-1, None))[0]:
            latest[name] = (int(steps), path)

    if bot_name:
        if bot_name not in latest:
            sys.exit(f"No checkpoints found for bot '{bot_name}' in checkpoints/.")
        return latest[bot_name][1] / "model.pt"
    if len(latest) > 1:
        sys.exit("checkpoints/ has more than one bot (" + ", ".join(sorted(latest)) + "). Pick one with --bot.")
    if not latest:
        return None
    return next(iter(latest.values()))[1] / "model.pt"

def has_model_config(model):
    """Whether the model stores the config it was trained with, which the bot builds its network and obs from."""
    try:
        with zipfile.ZipFile(model) as archive:
            return any(name.endswith("/data.pkl") and b"buta_config" in archive.read(name)
                       for name in archive.namelist())
    except zipfile.BadZipFile:
        sys.exit(f"{model} isn't a model file.")

def deploy():
    parser = argparse.ArgumentParser(description="Builds the RLBot bot and packages it into bot/.")
    parser.add_argument("checkpoint", nargs="?",
                        help="Checkpoint directory or .pt model file to deploy (e.g. a PFSP pool snapshot). "
                             "Defaults to the latest checkpoint in checkpoints/.")
    parser.add_argument("--bot", help="Bot name to take the latest checkpoint of, when checkpoints/ has several.")
    args = parser.parse_args()
    model = find_model(args.checkpoint, args.bot)
    if model and not has_model_config(model):
        sys.exit(f"{model} was saved before models stored their config, so the bot can't load it. Re-save it first by "
                 "running training with its checkpoint as the latest and 0 updates (see Deploying to RLBot in the README).")

    # LibTorch location, from the same environment variable the CMake build uses. Cargo's build script reads it too.
    libtorch = os.environ.get("LIBTORCH")
    if not libtorch or not (Path(libtorch) / "lib").is_dir():
        sys.exit("Set the LIBTORCH environment variable to your LibTorch directory (the one containing lib/ and share/).")
    torch_lib_dir = Path(libtorch) / "lib"

    print("Building buta_ppo_bot via Cargo...")
    subprocess.run(
        ["cargo", "build", "--release", "-p", "buta_ppo_bot"], 
        cwd="rust_bridge", 
        check=True
    )

    # libs/ only holds what this script copies, so start it empty to drop libraries an older deploy needed.
    bot_dir = Path("bot")
    bot_lib_dir = Path("bot/libs")
    shutil.rmtree(bot_lib_dir, ignore_errors=True)
    bot_lib_dir.mkdir(parents=True)

    print(f"Copying executable to {bot_dir.absolute()}...")
    shutil.copy("rust_bridge/target/release/buta_ppo_bot", bot_dir)

    # Cargo keeps a build directory per build script configuration, so older copies of the library can sit next to
    # the current one. The most recently built is the one the build above produced or kept.
    print("Locating custom C++ shared library...")
    libs = list(Path("rust_bridge/target/release/build").rglob("libbuta_ppo_bot.so"))
    if not libs:
        sys.exit("libbuta_ppo_bot.so not found in rust_bridge/target/release/build.")
    newest = max(libs, key=lambda p: p.stat().st_mtime)
    shutil.copy(newest, bot_lib_dir)
    built = datetime.fromtimestamp(newest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"  -> Copied {newest} (built {built})")

    # The library finds its dependencies next to itself, so copy every one ldd resolves into LibTorch, including
    # their own dependencies. System libraries (libc, libstdc++, ...) are left to the machine the bot runs on.
    print("Copying LibTorch dependencies...")
    ldd = subprocess.run(["ldd", str(bot_lib_dir / "libbuta_ppo_bot.so")], capture_output=True, text=True, check=True,
                         env={**os.environ, "LD_LIBRARY_PATH": str(torch_lib_dir)})
    for line in ldd.stdout.splitlines():
        name, arrow, rest = line.strip().partition(" => ")
        if not arrow:
            continue
        if rest.startswith("not found"):
            sys.exit(f"{name}, needed by libbuta_ppo_bot.so, not found in {torch_lib_dir}.")
        path = Path(rest.split(" (")[0])
        if path.resolve().is_relative_to(torch_lib_dir.resolve()):
            shutil.copy(path, bot_lib_dir / name)
            print(f"  -> Copied {name} ({path.stat().st_size / 2**20:.0f} MB)")

    # The model stores the network and obs settings it was trained with, so the bot needs nothing else from training.
    if model:
        shutil.copy(model, bot_dir / "model.pt")
        print(f"Copied {model} to {bot_dir / 'model.pt'}")
    else:
        print("Warning: no checkpoints found in checkpoints/. Copy a model.pt into bot/ yourself.")

    meshes = Path("assets/collision_meshes")
    if meshes.is_dir():
        shutil.copytree(meshes, bot_dir / "collision_meshes", dirs_exist_ok=True)
        print(f"Copied {meshes} to {bot_dir / 'collision_meshes'}")
    else:
        print(f"Warning: {meshes} not found. Copy your collision_meshes/ into bot/ yourself.")

    print("\nDeployment complete. Change the TEMPLATE values in bot/bot.toml and bot/loadout.toml for your bot.")

if __name__ == "__main__":
    deploy()