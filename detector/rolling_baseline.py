"""Tier 1: Adaptive rolling statistical baseline with Median Absolute Deviation (MAD)."""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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
        min_excess_sec: float = 0.12,
        adapt_after_consecutive: Optional[int] = None
    ):
        """
        Args:
            adapt_after_consecutive: If set, once a phase has been flagged this many
                cycles in a row the baseline treats the shift as the new normal and
                starts absorbing the elevated durations. None (default) never adapts,
                so persistent wear keeps alarming until rebaseline() is called.
        """
        self.window_size = window_size
        self.threshold_z = threshold_z
        self.min_samples_to_flag = min_samples_to_flag
        self.min_excess_sec = min_excess_sec
        self.adapt_after_consecutive = adapt_after_consecutive
        # Phase name -> deque of verified normal durations
        self.history: Dict[str, deque] = {}
        # Consecutively flagged anomalies counter to detect drift
        self.consecutive_anomalies: Dict[str, int] = {}

    def seed_nominal_baselines(self, nominal_timings: Dict[str, float]):
        """Pre-seeds the rolling buffer with nominal engineering values to eliminate cold-start distortion."""
        for name, nominal in nominal_timings.items():
            self.history[name] = deque([nominal] * (self.window_size // 2), maxlen=self.window_size)
            self.consecutive_anomalies[name] = 0

    def rebaseline(self, phase_name: str):
        """
        Discards a phase's baseline so it re-learns from the next observations.
        Call after an acknowledged permanent change (re-tooling, die change, repair).
        """
        self.history[phase_name] = deque(maxlen=self.window_size)
        self.consecutive_anomalies[phase_name] = 0

    def _compute_robust_stats(self, buffer: deque) -> Tuple[float, float, float]:
        """Calculates median, MAD, and robust standard deviation floor for duration buffer."""
        arr = np.array(buffer)
        median = float(np.median(arr))
        abs_deviations = np.abs(arr - median)
        mad = float(np.median(abs_deviations))

        sigma_robust = 1.4826 * mad
        if sigma_robust < 0.04:
            sigma_robust = max(0.04, float(np.std(arr)))

        return median, mad, sigma_robust

    def _classify_hypothesis(self, consec: int, excess_sec: float) -> str:
        """Determines the anomaly classification hypothesis based on duration and persistence."""
        if consec >= 4:
            return "CREEPING_WEAR"
        if excess_sec > 0.65:
            return "MICRO_STALL"
        return "VALVE_OVERLAP"

    def update_and_detect(
        self,
        cycle_id: int,
        phase_name: str,
        duration: float,
        **kwargs
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

        median, mad, sigma_robust = self._compute_robust_stats(buffer)
        z_score = (duration - median) / sigma_robust
        excess_sec = max(0.0, duration - (median + 1.645 * sigma_robust))
        is_anomaly = bool(z_score > self.threshold_z and excess_sec >= self.min_excess_sec)

        if is_anomaly:
            self.consecutive_anomalies[phase_name] += 1
            consec = self.consecutive_anomalies[phase_name]
            hypothesis = self._classify_hypothesis(consec, excess_sec)
            if self.adapt_after_consecutive is not None and consec >= self.adapt_after_consecutive:
                buffer.append(duration)
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
