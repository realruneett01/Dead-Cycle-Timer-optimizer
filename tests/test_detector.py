"""Unit tests for Tier 1, 2, and 3 anomaly and changepoint detectors."""
import numpy as np
from detector.cusum_detector import PageCusumDetector
from detector.rolling_baseline import RollingBaselineDetector
from detector.changepoint_detector import ChangepointDetector
from detector.hmm_detector import HMMDetector

def test_rolling_baseline_detection():
    """Tests rolling MAD baseline tracking and outlier detection."""
    np.random.seed(42)
    detector = RollingBaselineDetector(
        window_size=30, threshold_z=2.5, min_samples_to_flag=10, min_excess_sec=0.15
    )

    # Warmup with normal values (approx 2.5s with minor noise)
    for c in range(1, 20):
        val = 2.50 + float(np.random.normal(0, 0.03))
        res = detector.update_and_detect(cycle_id=c, phase_name="shear_stroke", duration=val)
        assert not res.is_anomaly

    # Inject a distinct micro-stall delay (+0.80s => 3.30s)
    stall_res = detector.update_and_detect(cycle_id=20, phase_name="shear_stroke", duration=3.35)
    assert stall_res.is_anomaly
    assert stall_res.excess_seconds > 0.50
    assert stall_res.robust_z_score > 3.0

def test_changepoint_detection():
    """Tests Pelt and Binseg offline changepoint localization on step shifts."""
    cp_detector = ChangepointDetector(min_size=5, penalty=2.0)

    # Flat signal: no changepoint
    flat = [2.0] * 30
    cps_flat = cp_detector.find_changepoints(flat)
    assert len(cps_flat) == 0

    # Step change signal: 20 samples at 2.0s, followed by 20 samples at 3.0s
    step = [2.0 + np.random.normal(0, 0.02) for _ in range(20)] + \
           [3.0 + np.random.normal(0, 0.02) for _ in range(20)]

    cps_step = cp_detector.find_changepoints(step)
    assert len(cps_step) >= 1
    # Detected changepoint should be near index 20
    assert any(17 <= cp <= 23 for cp in cps_step)

def test_hmm_detector():
    """Tests Gaussian HMM discrete state decoding with priors."""
    hmm_det = HMMDetector(n_states=3, random_state=42)
    hmm_det.initialize_with_priors(nominal_mean=2.5, nominal_std=0.08)

    # Normal durations should yield state 0
    normal_seq = [2.50, 2.52, 2.48, 2.51, 2.49]
    decoded_normal = hmm_det.decode_sequence(normal_seq)
    assert all(s == 0 for s in decoded_normal)

    # Stalled duration should decode to state 2 or state 1
    stalled_seq = [2.50, 2.51, 3.80]
    decoded_stall = hmm_det.decode_sequence(stalled_seq)
    assert decoded_stall[-1] in (1, 2)

def test_hmm_fit_orders_states_by_mean():
    """Tests that EM training monotonically orders states by duration mean."""
    rng = np.random.default_rng(0)
    durations = list(np.concatenate([
        rng.normal(2.5, 0.05, 60),
        rng.normal(3.0, 0.05, 20),
        rng.normal(3.75, 0.10, 10),
    ]))
    hmm_det = HMMDetector(n_states=3, random_state=42)
    hmm_det.fit(durations)

    means = hmm_det.model.means_.flatten()
    assert list(means) == sorted(means)
    decoded = hmm_det.decode_sequence(durations)
    assert decoded[0] == 0
    assert decoded[-1] == 2

def test_page_cusum_detector():
    """Tests sequential Page's CUSUM SPRT alarm triggering."""
    cusum = PageCusumDetector(
        alpha=0.01, beta=0.05, default_delta_sec=0.35, min_baseline_samples=10
    )

    # 15 nominal samples: should not trigger alarm
    for c in range(1, 16):
        res = cusum.evaluate_observation(
            cycle_id=c,
            phase_name="container_shift_close",
            duration=2.80 + float(np.random.normal(0, 0.04)),
            baseline_mu=2.80,
            baseline_sigma=0.08,
            sample_count=c
        )
        assert not res.is_alarm

    # Inject a sequence of delayed cycles (+0.40s)
    # CUSUM should accumulate evidence and assert alarm within 1-2 cycles
    alarm_fired = False
    for c in range(16, 20):
        res = cusum.evaluate_observation(
            cycle_id=c,
            phase_name="container_shift_close",
            duration=3.25,
            baseline_mu=2.80,
            baseline_sigma=0.08,
            sample_count=c
        )
        if res.is_alarm:
            alarm_fired = True
            assert res.cumulative_sum >= res.threshold_h or res.standardized_residual >= 4.0
            break

    assert alarm_fired, "Page CUSUM failed to trigger on consecutive +0.45s delays"

def test_rolling_baseline_rebaseline_after_permanent_shift():
    """Tests rebaseline capability after intentional mechanical change."""
    detector = RollingBaselineDetector(
        window_size=30, threshold_z=2.5, min_samples_to_flag=10, min_excess_sec=0.15
    )
    for c in range(1, 20):
        detector.update_and_detect(cycle_id=c, phase_name="die_slide_check", duration=1.80)

    # A permanent +0.6s shift (e.g. new die) keeps alarming against the old baseline
    res_shift = detector.update_and_detect(
        cycle_id=20, phase_name="die_slide_check", duration=2.40
    )
    assert res_shift.is_anomaly

    detector.rebaseline("die_slide_check")
    for c in range(21, 35):
        res = detector.update_and_detect(cycle_id=c, phase_name="die_slide_check", duration=2.40)
    assert not res.is_anomaly
    assert res.baseline_median == 2.40

def test_rolling_baseline_opt_in_adaptation():
    """Tests opt-in continuous adaptation after persistent consecutive flags."""
    detector = RollingBaselineDetector(
        window_size=20, threshold_z=2.5, min_samples_to_flag=10,
        min_excess_sec=0.15, adapt_after_consecutive=3
    )
    for c in range(1, 15):
        detector.update_and_detect(cycle_id=c, phase_name="billet_load", duration=3.20)

    flags = [
        detector.update_and_detect(cycle_id=c, phase_name="billet_load", duration=3.80).is_anomaly
        for c in range(15, 45)
    ]
    assert flags[0]
    assert not flags[-1], "Baseline should have absorbed the sustained shift"


def test_nan_and_sensor_glitch_protection():
    """Verifies that non-finite sensor values (NaN, Inf, <=0) do not poison buffers."""
    detector = RollingBaselineDetector(window_size=20, threshold_z=2.5, min_samples_to_flag=5)
    for c in range(1, 10):
        detector.update_and_detect(cycle_id=c, phase_name="shear_stroke", duration=2.5)

    # Ingest sensor glitch readings
    res_nan = detector.update_and_detect(cycle_id=10, phase_name="shear_stroke", duration=float("nan"))
    res_inf = detector.update_and_detect(cycle_id=11, phase_name="shear_stroke", duration=float("inf"))
    res_neg = detector.update_and_detect(cycle_id=12, phase_name="shear_stroke", duration=-1.5)

    assert not res_nan.is_anomaly
    assert not res_inf.is_anomaly
    assert not res_neg.is_anomaly
    assert res_nan.anomaly_type_hypothesis == "INVALID_READING"

    # Verify buffer was not contaminated and still computes clean finite median
    res_normal = detector.update_and_detect(cycle_id=13, phase_name="shear_stroke", duration=2.5)
    assert np.isfinite(res_normal.baseline_median)
    assert res_normal.baseline_median == 2.50


def test_detector_service_checkpoint_and_online_adaptation(tmp_path):
    """Verifies complete model persistence and online adaptation for industrial deployment."""
    from detector.detector_service import DetectorService, PhaseEventObservation

    ckpt_file = str(tmp_path / "model_ckpt.json")
    svc1 = DetectorService()

    # Process normal observations
    for c in range(1, 25):
        svc1.process_phase_event(PhaseEventObservation(cycle_id=c, phase_name="decompression", duration=2.2))

    # Save learned deployment state
    svc1.save_model_checkpoint(ckpt_file)
    svc1.close()

    # Create fresh unlearned service and restore checkpoint
    svc2 = DetectorService()
    svc2.load_model_checkpoint(ckpt_file)

    # Verify restored baseline
    res = svc2.process_phase_event(PhaseEventObservation(cycle_id=26, phase_name="decompression", duration=2.2))
    assert res["baseline_median"] == 2.20
    assert not res["is_anomaly"]

    # Test online regime adaptation (e.g. after tool/die change)
    new_regime_samples = [3.10 + float(np.random.normal(0, 0.02)) for _ in range(25)]
    svc2.adapt_to_new_regime("decompression", new_regime_samples)

    adapted_res = svc2.process_phase_event(PhaseEventObservation(cycle_id=27, phase_name="decompression", duration=3.10))
    assert abs(adapted_res["baseline_median"] - 3.10) < 0.05
    assert not adapted_res["is_anomaly"]

    # Test active learning operator feedback
    svc2.record_operator_feedback("decompression", 3.12, is_fault=False)
    assert svc2.tier1_baseline.consecutive_anomalies.get("decompression", 0) == 0
    svc2.close()


