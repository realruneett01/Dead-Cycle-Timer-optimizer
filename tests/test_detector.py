"""Unit tests for Tier 1, 2, and 3 anomaly and changepoint detectors."""
import pytest
import numpy as np
from detector.rolling_baseline import RollingBaselineDetector
from detector.changepoint_detector import ChangepointDetector
from detector.hmm_detector import HMMDetector

def test_rolling_baseline_detection():
    np.random.seed(42)
    detector = RollingBaselineDetector(window_size=30, threshold_z=2.5, min_samples_to_flag=10, min_excess_sec=0.15)
    
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

def test_page_cusum_detector():
    from detector.cusum_detector import PageCusumDetector
    cusum = PageCusumDetector(alpha=0.01, beta=0.05, default_delta_sec=0.35, min_baseline_samples=10)
    
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

