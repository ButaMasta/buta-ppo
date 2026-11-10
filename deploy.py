import argparse
import mmap
import os
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

# The Rust target of Windows bots, on every host.
WINDOWS_TARGET = "x86_64-pc-windows-msvc"
# Microsoft's link to the latest supported Visual C++ runtime installer, which the runtime DLLs are taken from.
VC_REDIST_URL = "https://aka.ms/vc14/vc_redist.x64.exe"

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

def host_os():
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        return "windows"
    sys.exit(f"Deploying from {sys.platform} isn't supported. Use Linux or Windows.")

def libtorch_version(libtorch):
    """The LibTorch release (e.g. 2.14.0) without its build suffix (+cu132, +cpu), or None if unknown."""
    try:
        return (Path(libtorch) / "build-version").read_text().strip().split("+")[0]
    except OSError:
        return None

def target_libtorch(target_os):
    """The LibTorch directory for the target: LIBTORCH_WINDOWS for Windows, falling back to LIBTORCH when the target is
    the host OS (bot/build.rs picks it the same way)."""
    same_os = target_os == host_os()
    names = (["LIBTORCH_WINDOWS"] if target_os == "windows" else []) + (["LIBTORCH"] if same_os else [])
    for name in names:
        value = os.environ.get(name)
        if value:
            if not (Path(value) / "lib").is_dir():
                sys.exit(f"{name} is {value}, which has no lib/ directory. Set it to the LibTorch directory (the one "
                         "containing lib/ and share/).")
            return Path(value)

    if target_os == "windows" and not same_os:
        version = libtorch_version(os.environ.get("LIBTORCH", "")) or "<version>"
        sys.exit("Set the LIBTORCH_WINDOWS environment variable to the Windows CPU LibTorch directory (the one "
                 "containing lib/ and share/). Use the LibTorch version you train with, e.g. "
                 f"https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-{version}%2Bcpu.zip")
    sys.exit("Set the LIBTORCH environment variable to your LibTorch directory (the one containing lib/ and share/).")

def xwin_cache_dir():
    """Where cargo-xwin keeps the Microsoft CRT and Windows SDK it downloads."""
    if os.environ.get("XWIN_CACHE_DIR"):
        return Path(os.environ["XWIN_CACHE_DIR"])
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cargo-xwin" / "xwin"

def check_cross_tools(accepted_license):
    """A Windows bot built on Linux needs cargo-xwin (clang-cl + lld-link with Microsoft's CRT and SDK) and the Windows
    Rust target. Its first build downloads Microsoft's files, which come under Microsoft's license."""
    if not shutil.which("cargo-xwin"):
        sys.exit("Building a Windows bot on Linux needs cargo-xwin, clang and lld: `cargo install --locked cargo-xwin`, "
                 "plus clang and lld from your package manager.")
    targets = subprocess.run(["rustup", "target", "list", "--installed"], capture_output=True, text=True)
    if targets.returncode != 0 or WINDOWS_TARGET not in targets.stdout.split():
        sys.exit(f"Building a Windows bot on Linux needs the {WINDOWS_TARGET} Rust target (and rustup): "
                 f"`rustup target add {WINDOWS_TARGET}`.")
    if not accepted_license and not xwin_cache_dir().is_dir():
        sys.exit("Building a Windows bot on Linux downloads Microsoft's C runtime and Windows SDK (via cargo-xwin) and "
                 "Microsoft's Visual C++ runtime DLLs, which the bot ships in libs/. They come under Microsoft's license "
                 "terms (https://go.microsoft.com/fwlink/?LinkId=2086102 and the Visual C++ redistributable's). Run "
                 "again with --accept-msvc-license to accept them.")

# ---------------------------------------------------------------------------------------------------------------------
# Windows DLL dependencies: which DLLs a DLL imports, read from its PE import tables.

def pe_imports(path):
    """The DLL names a PE file (DLL or EXE) imports, including delay-loaded ones."""
    with open(path, "rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        u16 = lambda offset: struct.unpack_from("<H", data, offset)[0]
        u32 = lambda offset: struct.unpack_from("<I", data, offset)[0]

        pe = u32(0x3C)
        if data[:2] != b"MZ" or data[pe:pe + 4] != b"PE\0\0":
            raise ValueError(f"{path} isn't a Windows DLL or executable.")
        coff = pe + 4
        section_count, optional_size = u16(coff + 2), u16(coff + 16)
        optional = coff + 20
        directories = optional + {0x10B: 96, 0x20B: 112}[u16(optional)]
        sections = [struct.unpack_from("<IIII", data, optional + optional_size + 40 * i + 8)
                    for i in range(section_count)]

        def offset_of(rva):
            for virtual_size, address, raw_size, raw_pointer in sections:
                if address <= rva < address + max(virtual_size, raw_size):
                    return rva - address + raw_pointer
            raise ValueError(f"{path}: address {rva:#x} is in no section.")

        def name_at(rva):
            start = offset_of(rva)
            return data[start:data.find(b"\0", start)].decode("ascii")

        names = []
        # Directory 1: import descriptors (20 bytes, DLL name at +12). Directory 13: delay-load descriptors (32 bytes,
        # DLL name at +4). Both end with an all-zero entry.
        for directory, entry_size, name_field in ((1, 20, 12), (13, 32, 4)):
            rva = u32(directories + 8 * directory)
            if rva == 0:
                continue
            entry = offset_of(rva)
            while (name_rva := u32(entry + name_field)) != 0:
                names.append(name_at(name_rva))
                entry += entry_size
        return names

# ---------------------------------------------------------------------------------------------------------------------
# Microsoft's Visual C++ runtime DLLs, taken from the official installer. Its payload is in cabinet (.cab) files, read
# here with zlib so no extraction tool is needed.

def cab_files(data, start=0):
    """The files in the cabinet at data[start:], as {name: bytes}. Supports uncompressed and MSZIP folders, which the
    Visual C++ installer uses."""
    u8 = lambda offset: data[start + offset]
    u16 = lambda offset: struct.unpack_from("<H", data, start + offset)[0]
    u32 = lambda offset: struct.unpack_from("<I", data, start + offset)[0]

    files_offset, folder_count, file_count, flags = u32(16), u16(26), u16(28), u16(30)
    header_reserve = folder_reserve = data_reserve = 0
    position = 36
    if flags & 4:  # Reserved areas present.
        header_reserve, folder_reserve, data_reserve = u16(36), u8(38), u8(39)
        position = 40 + header_reserve
    for flag in (1, 2):  # Names of a previous/next cabinet in a set: two strings each.
        if flags & flag:
            for _ in range(2):
                position = data.index(b"\0", start + position) - start + 1

    folders = []
    for _ in range(folder_count):
        data_offset, block_count, compression = struct.unpack_from("<IHH", data, start + position)
        folders.append((data_offset, block_count, compression & 0xF))
        position += 8 + folder_reserve

    def folder_bytes(index):
        data_offset, block_count, compression = folders[index]
        if compression not in (0, 1):
            raise ValueError(f"cabinet compression type {compression} isn't supported (only none and MSZIP).")
        out, history, block = [], b"", data_offset
        for _ in range(block_count):
            compressed_size = u16(block + 4)
            payload_start = start + block + 8 + data_reserve
            payload = data[payload_start:payload_start + compressed_size]
            if compression == 1:
                # MSZIP: "CK", then a deflate block that may refer back to the previous block's output.
                inflater = zlib.decompressobj(-15, zdict=history) if history else zlib.decompressobj(-15)
                payload = inflater.decompress(payload[2:]) + inflater.flush()
                history = payload[-32768:]
            out.append(payload)
            block += 8 + data_reserve + compressed_size
        return b"".join(out)

    contents, unpacked = {}, {}
    position = files_offset
    for _ in range(file_count):
        size, folder_offset, folder = struct.unpack_from("<IIH", data, start + position)
        name_end = data.index(b"\0", start + position + 16)
        name = data[start + position + 16:name_end].decode("utf-8", "replace")
        position = name_end - start + 1
        if folder not in unpacked:
            unpacked[folder] = folder_bytes(folder)
        contents[name] = unpacked[folder][folder_offset:folder_offset + size]
    return contents

def vc_runtime_dlls(cache_dir):
    """The x64 Visual C++ runtime DLLs ({lowercase name: path}): from MSVC_RUNTIME_DIR if set, else extracted from
    Microsoft's installer into cache_dir (downloaded once)."""
    if os.environ.get("MSVC_RUNTIME_DIR"):
        directory = Path(os.environ["MSVC_RUNTIME_DIR"])
    else:
        directory = cache_dir / "x64"
        if not (directory / "vcruntime140.dll").is_file():
            installer = cache_dir / "vc_redist.x64.exe"
            if not installer.is_file():
                print(f"Downloading the Visual C++ runtime from {VC_REDIST_URL}...")
                cache_dir.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(VC_REDIST_URL) as response:
                    installer.write_bytes(response.read())
            print("Extracting the Visual C++ runtime DLLs...")
            data = installer.read_bytes()
            # The installer holds cabinets: its own UI files, then the payload container, whose files are installer
            # databases and further cabinets. The x64 runtime DLLs are the files named <dll>_amd64 in those.
            dlls = {}
            start = data.find(b"MSCF")
            while start != -1:
                for inner in cab_files(data, start).values():
                    if inner[:4] == b"MSCF":
                        for name, content in cab_files(inner).items():
                            if name.lower().endswith(".dll_amd64"):
                                dlls[name[:-len("_amd64")]] = content
                start = data.find(b"MSCF", start + 4)
            if "vcruntime140.dll" not in {name.lower() for name in dlls}:
                sys.exit(f"Couldn't find the runtime DLLs in {installer}; Microsoft may have changed its format. Set "
                         "MSVC_RUNTIME_DIR to a folder with the x64 Visual C++ runtime DLLs (e.g. Visual Studio's "
                         r"VC\Redist\MSVC\<version>\x64\Microsoft.VC14x.CRT) instead.")
            directory.mkdir(parents=True, exist_ok=True)
            for name, content in dlls.items():
                (directory / name).write_bytes(content)
    return {path.name.lower(): path for path in directory.glob("*.dll")}

# ---------------------------------------------------------------------------------------------------------------------

def copy_linux_dependencies(bot_lib_dir, torch_lib_dir):
    """Copies every library the bot's libraries need from LibTorch, as ldd resolves them, including their own
    dependencies. System libraries (libc, libstdc++, ...) are left to the machine the bot runs on."""
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

def copy_windows_dependencies(bot_lib_dir, torch_lib_dir, runtime_dlls):
    """Copies every DLL the bot's DLLs need from LibTorch and the Visual C++ runtime, following their imports. The rest
    (KERNEL32.dll, api-ms-win-*.dll, ...) are part of Windows."""
    torch_dlls = {path.name.lower(): path for path in torch_lib_dir.glob("*.dll")}
    pending = [bot_lib_dir / "buta_ppo_bot.dll", bot_lib_dir / "rust_bridge.dll"]
    seen = {path.name.lower() for path in pending}
    system = set()
    while pending:
        for name in pe_imports(pending.pop()):
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            source = torch_dlls.get(key) or runtime_dlls.get(key)
            if source is None:
                if key.startswith(("vcruntime", "msvcp", "concrt", "vcomp")):
                    sys.exit(f"{name} is needed, but it isn't in the Visual C++ runtime that was found. LibTorch may "
                             "need a newer runtime than the installer's; set MSVC_RUNTIME_DIR to a newer one.")
                system.add(name)
                continue
            shutil.copy(source, bot_lib_dir / source.name)
            print(f"  -> Copied {source.name} ({source.stat().st_size / 2**20:.1f} MB)")
            pending.append(bot_lib_dir / source.name)
    print("  Left to Windows: " + ", ".join(sorted(system, key=str.lower)))

def deploy():
    # Line by line, so messages stay in order with the build and self-test output.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Builds the RLBot bot and packages it into bot/.")
    parser.add_argument("checkpoint", nargs="?",
                        help="Checkpoint directory or .pt model file to deploy (e.g. a PFSP pool snapshot). "
                             "Defaults to the latest checkpoint in checkpoints/.")
    parser.add_argument("--bot", help="Bot name to take the latest checkpoint of, when checkpoints/ has several.")
    parser.add_argument("--os", choices=["linux", "windows"], default=host_os(), dest="target_os",
                        help="The OS to build the bot for (default: this one). A Windows bot can be built on Linux; "
                             "it uses the Windows CPU LibTorch from LIBTORCH_WINDOWS.")
    parser.add_argument("--accept-msvc-license", action="store_true",
                        help="Accept the license terms of the Microsoft files a Windows build on Linux downloads.")
    parser.add_argument("--test", action="store_true",
                        help="Run the bot's self-test after deploying (with Wine for a Windows bot built on Linux).")
    args = parser.parse_args()
    target_os = args.target_os
    windows = target_os == "windows"
    cross = target_os != host_os()
    if cross and not windows:
        sys.exit("A Linux bot can only be built on Linux.")

    model = find_model(args.checkpoint, args.bot)
    if model and not has_model_config(model):
        sys.exit(f"{model} was saved before models stored their config, so the bot can't load it. Re-save it first by "
                 "running training with its checkpoint as the latest and 0 updates (see docs/deploying.md).")

    libtorch = target_libtorch(target_os)
    torch_lib_dir = libtorch / "lib"
    if cross:
        check_cross_tools(args.accept_msvc_license)
        # A model file is read by the bot's LibTorch, so match the version it was saved with.
        trained, deployed = libtorch_version(os.environ.get("LIBTORCH", "")), libtorch_version(libtorch)
        if trained and deployed and trained != deployed:
            print(f"Warning: training uses LibTorch {trained}, but LIBTORCH_WINDOWS is LibTorch {deployed}.")

    # Windows bots always name their target, so a Windows host's default Rust toolchain doesn't matter. The C++ library
    # links the Rust bridge library, which Cargo builds alongside the bot's build script rather than before it, so it's
    # built first (see bot/build.rs).
    cargo = ["cargo", "xwin", "build"] if cross else ["cargo", "build"]
    target_args = ["--target", WINDOWS_TARGET] if windows else []
    target_dir = Path("rust_bridge/target") / (WINDOWS_TARGET if windows else "") / "release"
    env = {**os.environ, "LIBTORCH_WINDOWS": str(libtorch)} if windows else os.environ
    for package in ("rust_bridge", "buta_ppo_bot"):
        print(f"Building {package} via Cargo for {target_os}...")
        if subprocess.run(cargo + ["--release", "-p", package] + target_args, cwd="rust_bridge", env=env).returncode:
            sys.exit(f"Building {package} failed.")

    # libs/ only holds what this script copies, so start it empty to drop libraries an older deploy needed.
    bot_dir = Path("bot")
    bot_lib_dir = Path("bot/libs")
    shutil.rmtree(bot_lib_dir, ignore_errors=True)
    bot_lib_dir.mkdir(parents=True)

    # bot/ holds one OS's build at a time, since libs/ does.
    exe_name, other_exe = ("buta_ppo_bot.exe", "buta_ppo_bot") if windows else ("buta_ppo_bot", "buta_ppo_bot.exe")
    print(f"Copying executable to {bot_dir.absolute()}...")
    shutil.copy(target_dir / exe_name, bot_dir)
    if (bot_dir / other_exe).exists():
        (bot_dir / other_exe).unlink()
        print(f"  -> Removed {bot_dir / other_exe} from an earlier deploy for the other OS")

    # Cargo keeps a build directory per build script configuration, so older copies of the library can sit next to
    # the current one. The most recently built is the one the build above produced or kept.
    lib_name, bridge_name = ("buta_ppo_bot.dll", "rust_bridge.dll") if windows else ("libbuta_ppo_bot.so",
                                                                                     "librust_bridge.so")
    print("Locating custom C++ shared library...")
    libs = list((target_dir / "build").rglob(lib_name))
    if not libs:
        sys.exit(f"{lib_name} not found in {target_dir / 'build'}.")
    newest = max(libs, key=lambda p: p.stat().st_mtime)
    shutil.copy(newest, bot_lib_dir)
    built = datetime.fromtimestamp(newest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"  -> Copied {newest} (built {built})")
    # The C++ library's own copy of the Rust bridge (the bot binary has another built in).
    shutil.copy(target_dir / bridge_name, bot_lib_dir)
    print(f"  -> Copied {target_dir / bridge_name}")

    if windows:
        print("Copying LibTorch and Visual C++ runtime DLLs...")
        copy_windows_dependencies(bot_lib_dir, torch_lib_dir, vc_runtime_dlls(Path("rust_bridge/target/msvc-runtime")))
    else:
        print("Copying LibTorch dependencies...")
        copy_linux_dependencies(bot_lib_dir, torch_lib_dir)

    # The model stores the network and obs settings it was trained with, so the bot needs nothing else from training.
    # bot/model.pt itself can be passed, to rebuild the bot and keep its model.
    if model and model.resolve() == (bot_dir / "model.pt").resolve():
        print(f"Kept {bot_dir / 'model.pt'}")
    elif model:
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

    if args.test:
        command = [str((bot_dir / exe_name).absolute()), "--self-test"]
        if cross:
            if not shutil.which("wine"):
                sys.exit("--test runs a Windows bot built on Linux with Wine, which wasn't found.")
            command = ["wine"] + command
        print(f"\nRunning the self-test: {' '.join(command)}")
        if subprocess.run(command, cwd=bot_dir, env={**os.environ, "WINEDEBUG": "-all"}).returncode:
            sys.exit("The self-test failed.")

    print("\nDeployment complete. Change the TEMPLATE values in bot/bot.toml and bot/loadout.toml for your bot.")

if __name__ == "__main__":
    deploy()
