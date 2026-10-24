"""Tests for commissioning checks, the plant gate, and the pendrive launcher helpers."""
import asyncio
import json
import subprocess
import sys

import numpy as np
import pytest

import launch
from deploy.commission import (
    analyse_signals, collect_samples, is_commissioned, report_path,
    validate_on_holdout, verify_against_profile,
)
from deploy.preflight import FAIL, PASS, WARN
from deploy.press_profile import profile_from_samples
from deploy.settings import Settings
from opcua.opcua_server import PressOpcUaServer
from simulator.press_state_machine import PressStateMachine


def _by_name(checks):
    return {c.name: c for c in checks}


def _stream(phase_values):
    out, cycle = [], 0
    for i in range(max(len(v) for v in phase_values.values())):
        cycle += 1
        for phase, vals in phase_values.items():
            if i < len(vals):
                out.append((cycle, phase, vals[i]))
    return out


def test_no_data_fails_liveness():
    assert analyse_signals([], 30)[0].status == FAIL


def test_frozen_sensor_is_detected():
    checks = _by_name(analyse_signals(_stream({"a": [2.0] * 40, "b": list(np.linspace(1, 1.2, 40))}), 30))
    assert checks["sensor-frozen:a"].status == FAIL


def test_unstable_sensor_and_bad_values():
    rng = np.random.default_rng(0)
    wild = list(rng.uniform(0.5, 6.0, 40))
    checks = _by_name(analyse_signals(_stream({"a": wild, "b": [1.0 + 0.01 * i for i in range(40)]}), 30))
    assert checks["sensor-stable:a"].status in (WARN, FAIL)
    bad = _by_name(analyse_signals([(1, "a", float("nan")), (2, "a", -1.0)] + _stream({"a": [1.0] * 30}), 30))
    assert bad["signal-valid"].status == FAIL


def test_stuck_cycle_counter_fails():
    samples = [(7, "a", 1.0 + 0.01 * i) for i in range(40)]
    assert _by_name(analyse_signals(samples, 30))["cycle-counter"].status == FAIL


def test_holdout_passes_on_healthy_and_flags_nonstationary():
    rng = np.random.default_rng(3)
    healthy = {"a": list(rng.normal(2.5, 0.1, 60)), "b": list(rng.normal(3.0, 0.1, 60))}
    s = Settings()
    stream = _stream(healthy)
    profile = profile_from_samples(healthy, learn_unknown=True)
    assert validate_on_holdout(stream, profile, s).status == PASS
    # Regime shifts +40% in the final stretch: must not pass silently.
    shifted = {k: v[:42] + [x * 1.4 for x in v[42:]] for k, v in healthy.items()}
    assert validate_on_holdout(_stream(shifted), profile, s).status in (WARN, FAIL)


def test_learn_unknown_phases_marks_heuristic_dead_cycle():
    rng = np.random.default_rng(1)
    prof = profile_from_samples({"clamp": list(rng.normal(4, 0.1, 40)),
                                 "stroke": list(rng.normal(70, 1, 40))}, learn_unknown=True)
    assert prof["phases"]["clamp"]["is_dead_cycle"] is True
    assert prof["phases"]["stroke"]["is_dead_cycle"] is False
    assert "confirm" in prof["phases"]["clamp"]["description"]


def test_verify_detects_wrong_machine(tmp_path):
    prof = {"phases": {"shear_stroke": {"nominal_sec": 2.5, "std_dev_sec": 0.1,
                                        "min_sec": 1.9, "max_sec": 4.0}}}
    path = tmp_path / "p.json"
    path.write_text(json.dumps(prof))
    s = Settings(press_profile=str(path))
    assert _by_name(verify_against_profile({"shear_stroke": [2.5] * 5}, s))["profile-match:shear_stroke"].status == PASS
    assert _by_name(verify_against_profile({"shear_stroke": [4.0] * 5}, s))["profile-match:shear_stroke"].status == FAIL
    assert _by_name(verify_against_profile({"mystery": [1.0] * 5}, s))["profile-phase:mystery"].status == FAIL


def test_gate_requires_passing_report_for_same_endpoint(tmp_path):
    s = Settings(data_dir=str(tmp_path), endpoint="opc.tcp://plant:4840")
    assert not is_commissioned(s)
    report_path(s).write_text(json.dumps({"passed": True, "endpoint": "opc.tcp://plant:4840"}))
    assert is_commissioned(s)
    s.endpoint = "opc.tcp://other:4840"
    assert not is_commissioned(s)
    report_path(s).write_text(json.dumps({"passed": False, "endpoint": "opc.tcp://other:4840"}))
    s.endpoint = "opc.tcp://other:4840"
    assert not is_commissioned(s)


def test_run_refuses_uncommissioned_plant(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "deploy", "run", "--endpoint", "opc.tcp://127.0.0.1:1/x"],
        capture_output=True, text=True, timeout=60, env={"DCTO_DATA_DIR": str(tmp_path)},
    )
    assert out.returncode == 2
    assert "commission" in out.stderr


def test_launcher_argument_routing():
    assert launch.default_args([]) == ["commission", "--run"]
    assert launch.default_args(["--demo"]) == ["commission", "--run", "--demo"]
    assert launch.default_args(["doctor"]) == ["doctor"]


def test_env_key_changes_with_requirements(monkeypatch, tmp_path):
    before = launch.env_key()
    (tmp_path / "requirements.txt").write_text("x==1\n")
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    assert launch.env_key() != before


async def _publish(server):
    machine, cycle = PressStateMachine(seed=1), 0
    while True:
        cycle += 1
        for event in machine.run_cycle(cycle_id=cycle):
            await server.publish_event(event)
            await asyncio.sleep(0.15)


@pytest.mark.asyncio
async def test_collect_samples_pairs_phase_and_duration_without_drops():
    endpoint = "opc.tcp://127.0.0.1:4846/freeopcua/server/"
    server = PressOpcUaServer(endpoint=endpoint)
    await server.init()
    async with server.server:
        task = asyncio.create_task(_publish(server))
        samples, dropped = await collect_samples(endpoint, {}, 3, 90)
        task.cancel()
    assert dropped == 0
    by_phase = {}
    for _, phase, dur in samples:
        by_phase.setdefault(phase, []).append(dur)
    nominal = PressStateMachine().config.phases
    for phase, vals in by_phase.items():
        assert abs(np.median(vals) - nominal[phase].nominal_sec) < 0.5 * nominal[phase].nominal_sec
