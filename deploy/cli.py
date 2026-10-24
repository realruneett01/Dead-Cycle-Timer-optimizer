"""`python -m deploy` entry point: doctor | configure | calibrate | commission | bundle | run | health."""
import argparse
import asyncio
import json
import logging
import platform
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from deploy.hostprobe import is_dir_writable, probe_host
from deploy.press_profile import calibrate_from_csv
from deploy.settings import DEFAULT_SETTINGS_PATH, PROJECT_ROOT, Settings, load_settings, save_settings
from opcua.opcua_nodes import build_opcua_endpoint

logger = logging.getLogger("DCTO")


def cmd_doctor(_args) -> int:
    """Prints host capabilities and the settings that would be applied; non-zero on blockers."""
    host = probe_host()
    settings = load_settings(host=host)
    blockers = []
    if host.missing_required:
        blockers.append(f"missing packages: {', '.join(host.missing_required)} (pip install -r requirements.txt)")
    if not host.numpy_compatible:
        blockers.append("Python >=3.13 has no wheels for numpy<2; use Python 3.11/3.12 or the Docker image")
    if not is_dir_writable(Path(settings.data_dir)):
        blockers.append(f"data_dir not writable: {settings.data_dir}")
    if settings.dashboard and host.missing_optional:
        blockers.append(f"dashboard enabled but missing: {', '.join(host.missing_optional)}")
    print(json.dumps({"host": host.to_dict(), "settings": settings.__dict__}, indent=2, default=str))
    for b in blockers:
        print(f"BLOCKER: {b}", file=sys.stderr)
    return 1 if blockers else 0


def cmd_configure(args) -> int:
    """Writes config/dcto.json tuned to this host (existing file is kept unless --force)."""
    dest = Path(args.output) if args.output else DEFAULT_SETTINGS_PATH
    if dest.exists() and not args.force:
        print(f"{dest} exists; use --force to regenerate")
        return 0
    settings = load_settings(path=Path("/nonexistent"))
    if args.endpoint:
        settings.endpoint = args.endpoint
    save_settings(settings, dest)
    print(f"Wrote {dest} (profile={settings.profile}, dashboard={settings.dashboard})")
    return 0


def cmd_calibrate(args) -> int:
    """Learns this press's phase timings from a healthy-run CSV and points settings at them."""
    profile = calibrate_from_csv(args.csv)
    if not profile["phases"]:
        print("No phase had enough samples to calibrate.", file=sys.stderr)
        return 1
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    settings = load_settings()
    settings.press_profile = str(dest)
    save_settings(settings)
    print(f"Calibrated {len(profile['phases'])} phase(s) -> {dest}")
    return 0


def cmd_commission(args) -> int:
    """Preflight + sensor discovery + live calibration + validation; optionally start running."""
    from deploy.commission import CommissionOptions, commission  # pylint: disable=import-outside-toplevel

    code = asyncio.run(commission(CommissionOptions(
        endpoint=args.endpoint or "", scan_subnet=args.scan, demo=args.demo,
        cycles=args.cycles, timeout_sec=args.timeout, verify=args.verify)))
    if code == 0 and args.run and not args.demo:
        return cmd_run(argparse.Namespace(endpoint=None, no_dashboard=args.no_dashboard,
                                          skip_commissioning=False))
    return code


def cmd_bundle(args) -> int:
    """Downloads wheels for this OS/Python into vendor/wheels so the stick installs offline."""
    dest = PROJECT_ROOT / "vendor" / "wheels"
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "-r", str(PROJECT_ROOT / "requirements.txt"),
           "-d", str(dest)]
    code = subprocess.call(cmd)
    if code == 0:
        tag = f"{platform.system()}-{platform.machine()}-py{sys.version_info.major}{sys.version_info.minor}"
        (dest / "BUNDLE_TARGET.txt").write_text(tag + "\n", encoding="utf-8")
        print(f"Wheelhouse built for {tag}. It only works on machines with the same OS, CPU and Python.")
    return code


def cmd_health(args) -> int:
    """Exit 0 when the detector wrote telemetry recently (container/systemd health probe)."""
    import time  # pylint: disable=import-outside-toplevel
    path = load_settings().telemetry_path
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        print(f"no telemetry at {path}", file=sys.stderr)
        return 1
    if age > args.max_age:
        print(f"telemetry stale: {age:.0f}s old", file=sys.stderr)
        return 1
    with open(path, encoding="utf-8") as f:
        rows = sum(1 for _ in f) - 1  # minus header
    if rows < 1:
        print("telemetry file has a header but no events yet", file=sys.stderr)
        return 1
    return 0


async def _watch_children(procs: List[subprocess.Popen], stop: asyncio.Event) -> bool:
    """Stops the run (so a supervisor can restart it) if the simulator or dashboard dies."""
    while not stop.is_set():
        for p in procs:
            if p.poll() is not None:
                logger.error("Child process %s exited with code %s; stopping", p.args[2:4], p.returncode)
                stop.set()
                return True
        await asyncio.sleep(1.0)
    return False


async def _run(settings: Settings) -> int:
    from deploy.runtime import run_ingest  # pylint: disable=import-outside-toplevel

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows event loop
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))

    procs: List[subprocess.Popen] = []
    endpoint = settings.endpoint
    try:
        if settings.use_simulator:
            endpoint = build_opcua_endpoint(settings.simulator_port)
            procs.append(subprocess.Popen([
                sys.executable, "-m", "opcua.opcua_server",
                "--port", str(settings.simulator_port),
                "--speedup", str(settings.simulator_speedup)], cwd=PROJECT_ROOT))
            await asyncio.sleep(2.0)
        if settings.dashboard:
            procs.append(subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "dashboard/dashboard_app.py",
                "--server.port", str(settings.dashboard_port),
                "--server.headless", "true", "--browser.gatherUsageStats", "false",
            ], cwd=PROJECT_ROOT))
        watchdog = asyncio.create_task(_watch_children(procs, stop))
        await run_ingest(settings, endpoint, stop)
        if watchdog.done() and watchdog.result():
            return 1
        watchdog.cancel()
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


def cmd_run(args) -> int:
    """Starts simulator (if no endpoint), dashboard (if profile allows) and the detector ingest."""
    settings = load_settings()
    if args.endpoint:
        settings.endpoint = args.endpoint
    if not settings.use_simulator and not args.skip_commissioning:
        from deploy.commission import is_commissioned  # pylint: disable=import-outside-toplevel
        if not is_commissioned(settings):
            print("This plant endpoint has not passed commissioning. Run: python -m deploy commission\n"
                  "(or --skip-commissioning to override at your own risk)", file=sys.stderr)
            return 2
    if args.no_dashboard:
        settings.dashboard = False
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    return asyncio.run(_run(settings))


def main(argv: Optional[List[str]] = None) -> int:
    """CLI dispatcher."""
    parser = argparse.ArgumentParser(prog="python -m deploy", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="probe host, show effective settings").set_defaults(fn=cmd_doctor)
    p = sub.add_parser("configure", help="generate config/dcto.json for this host")
    p.add_argument("--endpoint", help="opc.tcp:// URL of the real PLC/gateway")
    p.add_argument("--output")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_configure)
    p = sub.add_parser("calibrate", help="learn phase timings from a healthy-run CSV")
    p.add_argument("csv")
    p.add_argument("--output", default=str(PROJECT_ROOT / "config" / "press_profile.json"))
    p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("health", help="exit 0 if telemetry is fresh")
    p.add_argument("--max-age", type=float, default=120.0)
    p.set_defaults(fn=cmd_health)
    p = sub.add_parser("run", help="start the full stack for this host")
    p.add_argument("--endpoint")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--skip-commissioning", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("commission", help="check, discover sensors, calibrate and validate before go-live")
    p.add_argument("--endpoint")
    p.add_argument("--scan", action="store_true", help="also scan the local /24 for OPC-UA servers")
    p.add_argument("--demo", action="store_true", help="rehearse against the built-in simulator")
    p.add_argument("--cycles", type=int, default=30, help="cycles to learn from (minimum 43 so validation has data)")
    p.add_argument("--timeout", type=float, default=3600.0, help="max seconds to wait for data")
    p.add_argument("--verify", action="store_true",
                   help="quick check (5 cycles) against the saved calibration")
    p.add_argument("--run", action="store_true", help="start running if commissioning passes")
    p.add_argument("--no-dashboard", action="store_true")
    p.set_defaults(fn=cmd_commission)
    p = sub.add_parser("bundle", help="download wheels so this folder installs offline")
    p.set_defaults(fn=cmd_bundle)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("asyncua").setLevel(logging.ERROR)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
