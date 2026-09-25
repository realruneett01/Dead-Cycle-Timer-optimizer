"""End-to-end integration test verifying the complete DCTO pipeline."""
import pytest
from dashboard.oee_calculator import OEECalculator, PlantParameters
from detector.detector_service import DetectorService
from simulator.anomaly_injector import AnomalyInjector
from simulator.press_config import PressConfig
from simulator.press_state_machine import PressStateMachine

def test_full_pipeline_execution():
    """Verifies that simulator, anomaly injector, detector service, and OEE calculator integrate seamlessly."""
    config = PressConfig()
    injector = AnomalyInjector(anomaly_probability=0.20, seed=101)
    sm = PressStateMachine(config=config, injector=injector, seed=101)
    detector = DetectorService(config=config, threshold_z=2.75)
    calculator = OEECalculator(params=PlantParameters())

    total_events = 0
    total_anomalies = 0
    total_excess_sec = 0.0

    # Run 30 complete cycles
    for cycle_id in range(1, 31):
        events = sm.run_cycle(cycle_id=cycle_id)
        assert len(events) == 8, f"Cycle {cycle_id} did not yield 8 phases"
        total_events += len(events)

        for ev in events:
            result = detector.process_phase_event(
                cycle_id=ev.cycle_id,
                phase_name=ev.phase_name,
                duration=ev.duration_sec,
                is_dead_cycle=ev.is_dead_cycle
            )
            assert "is_anomaly" in result
            assert "actual_duration" in result
            assert "anomaly_type" in result

            if result["is_anomaly"]:
                total_anomalies += 1
                total_excess_sec += result["excess_seconds"]

    # Assert detector tracked events
    assert total_events == 240
    assert total_anomalies > 0, "No anomalies were detected across 30 cycles with p=0.20"

    # Evaluate OEE calculation
    metrics = calculator.calculate_recovery(
        total_excess_seconds=total_excess_sec,
        total_cycles=30,
        total_anomalies=total_anomalies,
        capture_efficiency=0.85
    )

    assert metrics.annual_recovered_hours > 0.0
    assert metrics.annual_direct_cost_savings_eur > 0.0
    assert metrics.annual_additional_tonnage_mt > 0.0
    assert metrics.oee_availability_gain_pct > 0.0
