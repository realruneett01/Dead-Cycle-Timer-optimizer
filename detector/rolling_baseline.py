"""Tier 1: Adaptive rolling statistical baseline with Median Absolute Deviation (MAD)."""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class DetectionResult:
    cycle_id: int
    phase_name: str
    actual_duration: float
    baseline_median: float
    baseline_mad: float
    robust_z_score: float
    is_anomaly: bool
    excess_seconds: float
    anomaly_type_hypothesis: str

class RollingBaselineDetector:
    """
    Online statistical detector tracking running window of phase durations.
    Uses Median and MAD to resist contamination from previously injected anomalies.
    """

    def __init__(
        self,
        window_size: int = 40,
        threshold_z: float = 2.65,
        min_samples_to_flag: int = 15,
        min_excess_sec: float = 0.12
    ):
        self.window_size = window_size
        self.threshold_z = threshold_z
        self.min_samples_to_flag = min_samples_to_flag
        self.min_excess_sec = min_excess_sec
        # Phase name -> deque of verified normal durations
        self.history: Dict[str, deque] = {}
        # Consecutively flagged anomalies counter to detect drift
        self.consecutive_anomalies: Dict[str, int] = {}

    def seed_nominal_baselines(self, nominal_timings: Dict[str, float]):
        """Pre-seeds the rolling buffer with nominal engineering values to eliminate cold-start distortion."""
        for name, nominal in nominal_timings.items():
            self.history[name] = deque([nominal] * (self.window_size // 2), maxlen=self.window_size)
            self.consecutive_anomalies[name] = 0

    def update_and_detect(
        self,
        cycle_id: int,
        phase_name: str,
        duration: float,
        is_dead_cycle: bool = True
    ) -> DetectionResult:
        """
        Ingests a new phase observation, evaluates against baseline,
        and conditionally updates the running buffer if within normal bounds.
        """
        if phase_name not in self.history:
            self.history[phase_name] = deque(maxlen=self.window_size)
            self.consecutive_anomalies[phase_name] = 0

        buffer = self.history[phase_name]

        # Cold-start warmup: accumulate initial samples
        if len(buffer) < self.min_samples_to_flag:
            buffer.append(duration)
            return DetectionResult(
                cycle_id=cycle_id,
                phase_name=phase_name,
                actual_duration=duration,
                baseline_median=duration,
                baseline_mad=0.0,
                robust_z_score=0.0,
                is_anomaly=False,
                excess_seconds=0.0,
                anomaly_type_hypothesis="WARMUP"
            )

        # Compute robust statistics
        arr = np.array(buffer)
        median = float(np.median(arr))
        abs_deviations = np.abs(arr - median)
        mad = float(np.median(abs_deviations))

        # Scale MAD to approximate standard deviation for normal distribution
        # If MAD is near zero or unphysically small, fallback to standard std dev with a realistic floor
        sigma_robust = 1.4826 * mad
        if sigma_robust < 0.04:
            sigma_robust = max(0.04, float(np.std(arr)))

        z_score = (duration - median) / sigma_robust
        excess_sec = max(0.0, duration - (median + 1.645 * sigma_robust))
        
        # Physical anomaly check: statistically anomalous AND exceeds minimum physical excess threshold
        is_anomaly = bool(z_score > self.threshold_z and excess_sec >= self.min_excess_sec)

        if is_anomaly:
            self.consecutive_anomalies[phase_name] += 1
            consec = self.consecutive_anomalies[phase_name]
            
            # Hypothesis classification
            if consec >= 4:
                hypothesis = "CREEPING_WEAR"
            elif excess_sec > 0.65:
                hypothesis = "MICRO_STALL"
            else:
                hypothesis = "VALVE_OVERLAP"
                
            # If an anomaly occurs, we do NOT poison the normal baseline buffer
            pass
        else:
            self.consecutive_anomalies[phase_name] = 0
            hypothesis = "NONE"
            buffer.append(duration)

        return DetectionResult(
            cycle_id=cycle_id,
            phase_name=phase_name,
            actual_duration=round(duration, 3),
            baseline_median=round(median, 3),
            baseline_mad=round(mad, 4),
            robust_z_score=round(z_score, 2),
            is_anomaly=is_anomaly,
            excess_seconds=round(excess_sec, 3),
            anomaly_type_hypothesis=hypothesis
        )
