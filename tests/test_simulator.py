"""Unit tests for Extrusion Press Simulator and Anomaly Injector."""
import os
import tempfile
import pandas as pd
import pytest
from simulator.anomaly_injector import AnomalyInjector, AnomalyType
from simulator.press_config import PressConfig
from simulator.press_state_machine import PressStateMachine

def test_press_config_nominal_values():
    config = PressConfig()
    assert len(config.canonical_phase_names) == 8
    assert len(config.dead_cycle_phase_names) == 7
    # Extrusion should not be dead cycle
    assert not config.phases["extrusion"].is_dead_cycle
    # Dead cycle duration should be approximately 18s
    assert 16.0 <= config.nominal_dead_cycle_duration <= 20.0
    # Total cycle duration should be approximately 83s
    assert 75.0 <= config.nominal_total_cycle_duration <= 90.0

def test_canonical_phase_order_and_bounds():
    config = PressConfig()
    sm = PressStateMachine(config=config, seed=123)
    
    expected_order = [
        "decompression",
        "container_shift_open",
        "shear_stroke",
        "die_slide",
        "billet_load",
        "container_shift_close",
        "rapid_advance",
        "extrusion"
    ]
    
    for cycle_id in range(1, 15):
        events = sm.run_cycle(cycle_id=cycle_id)
        assert len(events) == 8
        phase_names = [e.phase_name for e in events]
        assert phase_names == expected_order
        
        # Verify physical bounds
        for ev in events:
            assert ev.duration_sec > 0.5, f"Phase {ev.phase_name} duration too low: {ev.duration_sec}"
            assert 0.0 <= ev.pressure_bar <= 315.0
            assert 0.0 <= ev.valve_spool_pct <= 100.0

def test_anomaly_injection_and_ground_truth_logging():
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = os.path.join(tmpdir, "test_ground_truth.csv")
        injector = AnomalyInjector(anomaly_probability=0.30, ground_truth_path=gt_path, seed=42)
        sm = PressStateMachine(injector=injector, seed=42)
        
        events_with_anomaly = []
        for c in range(1, 51):
            events = sm.run_cycle(cycle_id=c)
            for ev in events:
                if ev.has_anomaly:
                    events_with_anomaly.append(ev)
                    
        assert os.path.exists(gt_path), "Ground truth file was not created"
        df_gt = pd.read_csv(gt_path)
        
        # Check that ground truth rows match the generated anomalies
        assert len(df_gt) == len(events_with_anomaly)
        assert len(df_gt) > 0, "No anomalies were generated with p=0.30 over 50 cycles"
        
        # Verify columns exist
        expected_cols = {
            "cycle_id", "timestamp", "phase_name", "anomaly_type",
            "injected_delay_sec", "nominal_duration", "actual_duration"
        }
        assert expected_cols.issubset(set(df_gt.columns))
        
        # Injected delays must be strictly positive
        assert (df_gt["injected_delay_sec"] > 0).all()
        
        # Verify anomaly types are valid
        valid_types = {AnomalyType.VALVE_OVERLAP.value, AnomalyType.MICRO_STALL.value, AnomalyType.CREEPING_WEAR.value}
        assert set(df_gt["anomaly_type"].unique()).issubset(valid_types)

def test_determinism_with_seed():
    config = PressConfig()
    with tempfile.TemporaryDirectory() as tmpdir:
        gt1 = os.path.join(tmpdir, "gt1.csv")
        gt2 = os.path.join(tmpdir, "gt2.csv")
        
        inj1 = AnomalyInjector(anomaly_probability=0.20, ground_truth_path=gt1, seed=999)
        sm1 = PressStateMachine(config=config, injector=inj1, seed=999)
        
        inj2 = AnomalyInjector(anomaly_probability=0.20, ground_truth_path=gt2, seed=999)
        sm2 = PressStateMachine(config=config, injector=inj2, seed=999)
        
        for c in range(1, 10):
            ev1 = sm1.run_cycle(cycle_id=c)
            ev2 = sm2.run_cycle(cycle_id=c)
            for e1, e2 in zip(ev1, ev2):
                assert e1.phase_name == e2.phase_name
                assert e1.duration_sec == e2.duration_sec
                assert e1.has_anomaly == e2.has_anomaly
                assert e1.injected_delay_sec == e2.injected_delay_sec
