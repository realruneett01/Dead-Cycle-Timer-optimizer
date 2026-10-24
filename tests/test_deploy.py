"""Tests for host probing, profile selection, settings layering, calibration, and node discovery."""
import csv
import json

import numpy as np
import pytest
from asyncua import Client

from deploy.hostprobe import HostInfo, first_free_port, is_port_free, probe_host
from deploy.press_profile import calibrate_from_csv, load_press_config
from deploy.profiles import EDGE, SERVER, STANDARD, select_profile
from deploy.runtime import discover_nodes
from deploy.settings import load_settings
from opcua.opcua_server import PressOpcUaServer


def _host(cpus=4, mem=8192, arch="x86_64") -> HostInfo:
    return HostInfo("Linux", arch, cpus, mem, "3.12.0", False)


def test_profile_selection_matches_hardware():
    assert select_profile(_host(cpus=2)) is EDGE
    assert select_profile(_host(mem=1024)) is EDGE
    assert select_profile(_host(arch="armv7l", cpus=4)) is EDGE
    assert select_profile(_host(cpus=4, mem=8192)) is STANDARD
    assert select_profile(_host(cpus=16, mem=32768)) is SERVER
    assert select_profile(_host(cpus=4, mem=0)) is STANDARD  # unknown memory is not "edge"


def test_probe_host_reports_sane_values():
    info = probe_host()
    assert info.cpu_count >= 1
    assert info.arch
    assert "numpy" not in info.missing_required


def test_first_free_port_skips_busy_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert not is_port_free(busy)
        assert first_free_port(busy) != busy


def test_settings_layering_env_over_file_over_defaults(tmp_path, monkeypatch):
    cfg = tmp_path / "dcto.json"
    cfg.write_text(json.dumps({"threshold_z": 3.0, "dashboard": False}))
    monkeypatch.setenv("DCTO_THRESHOLD_Z", "3.25")
    monkeypatch.setenv("DCTO_ENDPOINT", "opc.tcp://10.0.0.5:4840")
    s = load_settings(cfg, host=_host())
    assert s.threshold_z == 3.25
    assert s.dashboard is False
    assert s.endpoint == "opc.tcp://10.0.0.5:4840"
    assert s.use_simulator is False  # a real endpoint disables the simulator


def test_settings_reject_unknown_keys(tmp_path):
    cfg = tmp_path / "dcto.json"
    cfg.write_text(json.dumps({"threshold_zz": 3.0}))
    with pytest.raises(ValueError, match="threshold_zz"):
        load_settings(cfg, host=_host())


def test_calibration_learns_phase_timing_and_ignores_outliers(tmp_path):
    rng = np.random.default_rng(1)
    rows = [("shear_stroke", float(x)) for x in rng.normal(2.9, 0.1, 200)]
    rows += [("shear_stroke", 9.0)] * 3  # faults must not skew the baseline
    rows += [("billet_load", 3.0)] * 5   # too few samples: keep defaults
    src = tmp_path / "log.csv"
    with open(src, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle_id", "phase_name", "actual_duration"])
        for i, (p, d) in enumerate(rows):
            w.writerow([i, p, d])
    profile = calibrate_from_csv(str(src))
    assert set(profile["phases"]) == {"shear_stroke"}
    assert profile["phases"]["shear_stroke"]["nominal_sec"] == pytest.approx(2.9, abs=0.05)

    path = tmp_path / "press.json"
    path.write_text(json.dumps(profile))
    cfg = load_press_config(str(path))
    assert cfg.phases["shear_stroke"].nominal_sec == pytest.approx(2.9, abs=0.05)
    assert cfg.phases["billet_load"].nominal_sec == 3.2  # untouched default


def test_press_profile_rejects_unknown_phase_field(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"phases": {"shear_stroke": {"nominal": 2.0}}}))
    with pytest.raises(ValueError, match="nominal"):
        load_press_config(str(path))


@pytest.mark.asyncio
async def test_node_discovery_finds_nodes_without_hardcoded_paths():
    endpoint = "opc.tcp://127.0.0.1:4843/freeopcua/server/"
    server = PressOpcUaServer(endpoint=endpoint)
    await server.init()
    async with server.server:
        async with Client(url=endpoint) as client:
            nodes = await discover_nodes(client, {})
            assert set(nodes) == {"phase", "cycle", "duration"}
            with pytest.raises(LookupError, match="node_names"):
                await discover_nodes(client, {"duration": "NoSuchNode"})


def test_cgroup_cpu_quota_caps_cpu_count(monkeypatch):
    from deploy import hostprobe
    monkeypatch.setattr(hostprobe, "_cgroup_cpu_limit", lambda: 1.5)
    assert hostprobe._cpu_count() <= 2
    monkeypatch.setattr(hostprobe, "_cgroup_cpu_limit", lambda: None)
    assert hostprobe._cpu_count() >= 1
