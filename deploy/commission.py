"""Live commissioning: preflight -> find sensors -> collect -> signal checks -> calibrate -> validate.

Writes a machine-readable report that `run` consults before talking to a real plant.
"""
import asyncio
import json
import logging
import math
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from asyncua import Client

from deploy.discovery import find_endpoints, local_subnet, subnet_hosts
from deploy.hostprobe import HostInfo, first_free_port, is_port_free, probe_host
from deploy.preflight import FAIL, PASS, WARN, Check, run_preflight
from deploy.press_profile import MAD_TO_SIGMA, MIN_CALIBRATION_SAMPLES, load_press_config, profile_from_samples
from deploy.runtime import TelemetryTap, discover_nodes
from deploy.settings import PROJECT_ROOT, Settings, load_settings, save_settings
from detector.detector_service import DetectorService, DetectorServiceConfig, PhaseEventObservation
from opcua.opcua_nodes import build_opcua_endpoint

logger = logging.getLogger("Commission")

REPORT_NAME = "commissioning_report.json"
HOLDOUT_FRACTION = 0.3
SKIPPED_WARN = 0.02
ALARM_RATE_WARN = 0.05
ALARM_RATE_FAIL = 0.15
UNSTABLE_CV_WARN = 0.25
UNSTABLE_CV_FAIL = 0.50
FROZEN_MIN_SAMPLES = 10
BAD_VALUE_FAIL_FRACTION = 0.01
DEFAULT_MODEL_DEVIATION = 0.5
SAMPLE_WAIT_SEC = 3600.0
VERIFY_DEVIATION_WARN = 0.10
VERIFY_DEVIATION_FAIL = 0.25
VERIFY_MIN_CYCLES = 5

Sample = Tuple[int, str, float]


@dataclass
class CommissionOptions:
    """CLI-controlled commissioning parameters."""
    endpoint: str = ""
    scan_subnet: bool = False
    demo: bool = False
    cycles: int = MIN_CALIBRATION_SAMPLES
    timeout_sec: float = SAMPLE_WAIT_SEC
    assume_yes: bool = False
    verify: bool = False  # short check against the saved profile instead of full recalibration


def report_path(settings: Settings) -> Path:
    """Location of the commissioning report."""
    return Path(settings.data_dir) / REPORT_NAME


def is_commissioned(settings: Settings) -> bool:
    """True when a passing report exists for the currently configured endpoint."""
    try:
        report = json.loads(report_path(settings).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(report.get("passed")) and report.get("endpoint") == settings.endpoint


async def collect_samples(
    endpoint: str,
    overrides: Dict[str, str],
    min_per_phase: int,
    timeout_sec: float,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[Sample], int]:
    """Returns (samples, number of snapshots dropped because updates were skipped)."""
    """Subscribes to PhaseDuration and records (cycle, phase, duration) until enough cycles arrive."""
    samples: List[Sample] = []
    async with Client(url=endpoint, timeout=5) as client:
        tap = TelemetryTap(await discover_nodes(client, overrides))
        sub = await tap.start(client)
        deadline = time.monotonic() + timeout_sec
        try:
            while time.monotonic() < deadline:
                try:
                    samples.append(await asyncio.wait_for(tap.queue.get(), timeout=min(30.0, timeout_sec)))
                except asyncio.TimeoutError:
                    continue
                counts = Counter(p for _, p, _ in samples)
                if progress:
                    progress(len(samples), min(counts.values()))
                if len(counts) >= 2 and min(counts.values()) >= min_per_phase:
                    break
        finally:
            await sub.delete()
        dropped = tap.dropped
    return samples, dropped


def analyse_signals(samples: List[Sample], min_per_phase: int) -> List[Check]:
    """Sanity-checks the live stream: liveness, bad values, frozen/unstable sensors, coverage."""
    checks: List[Check] = []
    if not samples:
        return [Check("signal-liveness", FAIL, "no PhaseDuration updates received",
                      "Check the cycle is running and the sensor/gateway publishes changes")]
    checks.append(Check("signal-liveness", PASS, f"{len(samples)} updates received"))

    values = np.array([d for _, _, d in samples], dtype=float)
    bad = int(np.sum(~np.isfinite(values) | (values <= 0.0)))
    frac = bad / len(values)
    status = PASS if bad == 0 else (WARN if frac <= BAD_VALUE_FAIL_FRACTION else FAIL)
    checks.append(Check("signal-valid", status, f"{bad}/{len(values)} NaN or non-positive durations",
                        "Check units (seconds expected) and sensor wiring" if bad else ""))

    by_phase: Dict[str, List[float]] = {}
    for _, phase, dur in samples:
        if np.isfinite(dur) and dur > 0.0:
            by_phase.setdefault(phase, []).append(dur)

    if len({c for c, _, _ in samples}) < 2:
        checks.append(Check("cycle-counter", FAIL, "CycleNumber never advanced",
                            "Map node_names.cycle to the real cycle counter"))
    short = {p: len(v) for p, v in by_phase.items() if len(v) < min_per_phase}
    checks.append(Check(
        "phase-coverage", PASS if by_phase and not short else FAIL,
        f"{len(by_phase)} phase(s) seen" + (f"; too few samples: {short}" if short else ""),
        "Let more cycles run (--cycles / --timeout)" if short else ""))

    for phase, vals in sorted(by_phase.items()):
        arr = np.array(vals)
        median = float(np.median(arr))
        cv = MAD_TO_SIGMA * float(np.median(np.abs(arr - median))) / median
        if len(arr) >= FROZEN_MIN_SAMPLES and float(np.std(arr)) == 0.0:
            checks.append(Check(f"sensor-frozen:{phase}", FAIL, f"{len(arr)} identical readings",
                                "Sensor is likely stuck or the value is a constant"))
        elif cv > UNSTABLE_CV_FAIL:
            checks.append(Check(f"sensor-stable:{phase}", FAIL, f"robust CV {cv:.0%}",
                                "Timing too erratic to baseline; check sensor and phase mapping"))
        elif cv > UNSTABLE_CV_WARN:
            checks.append(Check(f"sensor-stable:{phase}", WARN, f"robust CV {cv:.0%}",
                                "Unusually noisy; calibrate during steady production"))
    return checks


def verify_against_profile(by_phase: Dict[str, List[float]], settings: Settings) -> List[Check]:
    """Compares live medians with the saved profile; catches a different machine or wrong mapping."""
    saved = load_press_config(settings.press_profile)
    checks: List[Check] = []
    for name, vals in sorted(by_phase.items()):
        if name not in saved.phases:
            checks.append(Check(f"profile-phase:{name}", FAIL, "phase not in saved profile",
                                "Plant changed? run commission without --verify"))
            continue
        ref = saved.phases[name].nominal_sec
        dev = abs(float(np.median(vals)) - ref) / ref
        status = PASS if dev <= VERIFY_DEVIATION_WARN else (WARN if dev <= VERIFY_DEVIATION_FAIL else FAIL)
        checks.append(Check(f"profile-match:{name}", status,
                            f"live median {np.median(vals):.2f}s vs calibrated {ref:.2f}s ({dev:.0%})",
                            "Recalibrate (commission without --verify) if the press was re-tooled"
                            if status != PASS else ""))
    return checks


def compare_to_default_model(profile: Dict, settings: Settings) -> List[Check]:
    """Flags learned phases that differ strongly from the built-in 28 MN model (informational)."""
    base = load_press_config(settings.press_profile or None)
    checks: List[Check] = []
    novel = [p for p in profile["phases"] if p not in base.phases]
    if novel:
        checks.append(Check("new-phases", WARN, f"auto-discovered: {', '.join(novel)}",
                            "Confirm dead-cycle vs productive flags in config/press_profile.json"))
    for name, spec in profile["phases"].items():
        if name in base.phases:
            ref = base.phases[name].nominal_sec
            if abs(spec["nominal_sec"] - ref) / ref > DEFAULT_MODEL_DEVIATION:
                checks.append(Check(
                    f"differs-from-default:{name}", WARN,
                    f"learned {spec['nominal_sec']}s vs built-in {ref}s",
                    "Expected on a different press; verify the phase mapping is correct"))
    return checks


def validate_on_holdout(samples: List[Sample], profile: Dict, settings: Settings) -> Check:
    """Calibrates on the early part, then measures the fused detector's alarm rate on the last 30%."""
    cut = int(len(samples) * (1.0 - HOLDOUT_FRACTION))
    train: Dict[str, List[float]] = {}
    for _, phase, dur in samples[:cut]:
        train.setdefault(phase, []).append(dur)
    train_profile = profile_from_samples(train, learn_unknown=True)
    if not train_profile["phases"]:
        return Check("holdout-validation", FAIL, "too little data to hold out a validation set",
                     "Collect more cycles")
    with tempfile.TemporaryDirectory() as tmp:
        prof = Path(tmp) / "p.json"
        prof.write_text(json.dumps(train_profile), encoding="utf-8")
        service = DetectorService(DetectorServiceConfig(
            config=load_press_config(str(prof)),
            telemetry_log_path=str(Path(tmp) / "t.csv"),
            threshold_z=settings.threshold_z,
            cusum_alpha=settings.cusum_alpha,
            cusum_beta=settings.cusum_beta,
        ))
        flagged = total = 0
        try:
            for i, (cycle, phase, dur) in enumerate(samples):
                rec = service.process_phase_event(PhaseEventObservation(cycle, phase, dur))
                if i >= cut:
                    total += 1
                    flagged += int(rec["is_anomaly"])
        finally:
            service.close()
    rate = flagged / max(total, 1)
    status = PASS if rate <= ALARM_RATE_WARN else (WARN if rate <= ALARM_RATE_FAIL else FAIL)
    return Check("holdout-validation", status,
                 f"{flagged}/{total} held-out events flagged ({rate:.1%})",
                 "High alarm rate on supposedly normal data: the plant may already be faulty "
                 "or not steady; recalibrate during healthy production" if status != PASS else "")


async def _resolve_endpoint(opts: CommissionOptions, settings: Settings) -> Tuple[Optional[str], List[Check]]:
    explicit = opts.endpoint or settings.endpoint
    if explicit:
        return explicit, [Check("endpoint", PASS, f"using {explicit}")]
    hosts = ["127.0.0.1"]
    if opts.scan_subnet:
        net = local_subnet()
        hosts = subnet_hosts(net) if net else hosts
    found = await find_endpoints(hosts)
    if len(found) == 1:
        return found[0], [Check("endpoint", PASS, f"discovered {found[0]}")]
    if not found:
        return None, [Check("endpoint", FAIL, "no OPC-UA server found",
                            "Pass --endpoint opc.tcp://host:port, or --scan to search the local /24")]
    return None, [Check("endpoint", FAIL, f"several servers found: {', '.join(found)}",
                        "Pick one with --endpoint")]


async def _wait_for_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_port_free(port):
            return True
        await asyncio.sleep(0.25)
    return False


def _worst(checks: List[Check]) -> str:
    statuses = {c.status for c in checks}
    return FAIL if FAIL in statuses else (WARN if WARN in statuses else PASS)


def _print_report(checks: List[Check], passed: bool) -> None:
    icon = {PASS: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}
    for c in checks:
        print(f"{icon[c.status]} {c.name}: {c.detail}")
        if c.fix:
            print(f"        -> {c.fix}")
    print("\nCOMMISSIONING " + ("PASSED" if passed else "FAILED - plant connection not enabled"))


def _write_report(settings: Settings, endpoint: str, host: HostInfo, checks: List[Check], passed: bool):
    dest = report_path(settings)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "passed": passed,
        "endpoint": endpoint,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": host.to_dict(),
        "checks": [c.to_dict() for c in checks],
    }, indent=2) + "\n", encoding="utf-8")


async def commission(opts: CommissionOptions) -> int:
    """Runs the full pipeline; returns 0 only when the plant connection is cleared for use."""
    host = probe_host()
    settings = load_settings(host=host)
    if opts.verify:
        if not (settings.press_profile and Path(settings.press_profile).exists()):
            print("No saved calibration to verify against; run a full commission first.")
            return 1
        opts.cycles = min(opts.cycles, VERIFY_MIN_CYCLES)
    else:
        # Enough that the 70% calibration share still meets the per-phase minimum.
        opts.cycles = max(opts.cycles, math.ceil(MIN_CALIBRATION_SAMPLES / (1.0 - HOLDOUT_FRACTION)))
    checks = run_preflight(host, settings)
    demo_proc: Optional[subprocess.Popen] = None
    endpoint: Optional[str] = None
    try:
        if _worst(checks) != FAIL:
            if opts.demo:
                port = first_free_port(4860)
                demo_proc = subprocess.Popen(
                    [sys.executable, "-m", "opcua.opcua_server", "--port", str(port), "--speedup", "40"],
                    cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if not await _wait_for_port(port):
                    checks.append(Check("demo-server", FAIL, "simulator did not start", "Run doctor"))
                opts.endpoint = build_opcua_endpoint(port)
            endpoint, ep_checks = await _resolve_endpoint(opts, settings)
            checks += ep_checks
        if endpoint and _worst(checks) != FAIL:
            checks += await _live_stage(opts, settings, endpoint)
    finally:
        if demo_proc:
            demo_proc.terminate()
            demo_proc.wait(timeout=10)

    passed = _worst(checks) != FAIL
    _print_report(checks, passed)
    if endpoint and not opts.demo:
        _write_report(settings, endpoint, host, checks, passed)
    elif opts.demo:
        print("(demo run against the built-in simulator: nothing saved, plant gate unchanged)")
    return 0 if passed else 1


async def _live_stage(opts: CommissionOptions, settings: Settings, endpoint: str) -> List[Check]:
    last = [0]

    def progress(total: int, minimum: int):
        if minimum != last[0]:
            last[0] = minimum
            print(f"  collected {minimum}/{opts.cycles} cycles ({total} updates)", flush=True)

    print(f"Collecting {opts.cycles} cycles from {endpoint} (up to {opts.timeout_sec:.0f}s)...")
    try:
        samples, dropped = await collect_samples(endpoint, settings.node_names, opts.cycles,
                                        opts.timeout_sec, progress)
    except LookupError as exc:
        return [Check("node-discovery", FAIL, str(exc), "Set node_names in config/dcto.json")]
    except (OSError, asyncio.TimeoutError) as exc:
        return [Check("connect", FAIL, f"{exc}", "Check network, firewall, and endpoint URL")]
    checks = [Check("connect", PASS, endpoint), Check("node-discovery", PASS, "phase/cycle/duration found")]
    checks += analyse_signals(samples, opts.cycles)
    lost = dropped / max(len(samples) + dropped, 1)
    checks.append(Check(
        "sampling-integrity",
        PASS if dropped == 0 else (WARN if lost <= SKIPPED_WARN else FAIL),
        f"{dropped} of {len(samples) + dropped} updates skipped ({lost:.1%})",
        "Source publishes faster than the sampling interval; data would be mislabelled. "
        "Slow the source or lower SAMPLING_MS" if dropped else ""))
    if _worst(checks) == FAIL:
        return checks

    by_phase: Dict[str, List[float]] = {}
    for _, phase, dur in samples:
        by_phase.setdefault(phase, []).append(dur)
    if opts.verify:
        checks += verify_against_profile(by_phase, settings)
        if _worst(checks) != FAIL:
            settings.endpoint = endpoint
            save_settings(settings)
        return checks
    profile = profile_from_samples(by_phase, learn_unknown=True)
    checks += compare_to_default_model(profile, settings)
    checks.append(validate_on_holdout(samples, profile, settings))

    if _worst(checks) != FAIL and not opts.demo:
        dest = PROJECT_ROOT / "config" / "press_profile.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        settings.endpoint = endpoint
        settings.press_profile = str(dest)
        save_settings(settings)
        checks.append(Check("profile-saved", PASS, f"{len(profile['phases'])} phase(s) -> {dest}"))
    return checks
