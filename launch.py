"""Pendrive launcher (stdlib only): prepares a per-machine environment, then commissions and runs.

Run via dcto.sh / dcto.bat. The environment is built in the host's user cache directory, never on the
stick, because FAT/exFAT sticks cannot hold a working venv. Calibration, config, and telemetry stay
on the stick, so the stick carries the plant's learned state between computers.
"""
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WHEELS = ROOT / "vendor" / "wheels"
SUPPORTED = ((3, 11), (3, 12))  # requirements.txt pins numpy<2


def cache_dir() -> Path:
    """Per-user cache location for the virtual environment."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "dcto"


def env_key() -> str:
    """Environment identity: changes with requirements, Python, OS, or CPU."""
    req = (ROOT / "requirements.txt").read_bytes()
    raw = req + f"{sys.version_info[:2]}{platform.system()}{platform.machine()}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def venv_python(venv: Path) -> Path:
    """Interpreter path inside a venv."""
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_env() -> Path:
    """Creates and populates the venv on first use; returns its interpreter."""
    venv = cache_dir() / f"venv-{env_key()}"
    marker = venv / ".dcto_ready"
    if marker.exists():
        return venv_python(venv)
    print(f"First run on this computer: preparing environment in {venv}")
    subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
    py = str(venv_python(venv))
    pip = [py, "-m", "pip", "install", "--quiet", "-r", str(ROOT / "requirements.txt")]
    if WHEELS.exists() and any(WHEELS.glob("*.whl")):
        print("Installing from the wheelhouse on the stick (offline).")
        pip += ["--no-index", "--find-links", str(WHEELS)]
    else:
        print("No wheelhouse on the stick; installing from the internet.")
    if subprocess.call(pip) != 0:
        sys.exit(
            "Package install failed. If this computer is offline, build the stick with "
            "`python -m deploy bundle` on a machine with the same OS, CPU and Python version."
        )
    marker.write_text("ok\n", encoding="utf-8")
    return venv_python(venv)


def default_args(argv):
    """No args => full guided flow; bare flags extend it; a subcommand passes through."""
    if not argv:
        return ["commission", "--run"]
    if argv[0].startswith("-"):
        return ["commission", "--run"] + argv
    return argv


def saved_calibration_exists() -> bool:
    """True when the stick already carries a calibrated profile and endpoint."""
    return (ROOT / "config" / "press_profile.json").exists() and (ROOT / "config" / "dcto.json").exists()


def main() -> int:
    """Entry point."""
    if sys.version_info[:2] not in SUPPORTED:
        sys.exit(f"Python {platform.python_version()} is unsupported; need 3.11 or 3.12.")
    py = ensure_env()
    args = default_args(sys.argv[1:])
    if args[:2] == ["commission", "--run"] and "--verify" not in args and "--demo" not in args \
            and saved_calibration_exists():
        args.append("--verify")
        print("Saved calibration found: running quick verification (use `commission` to recalibrate).")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.call([str(py), "-m", "deploy"] + args, cwd=ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main())
