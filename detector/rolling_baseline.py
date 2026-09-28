"""Tier 1: Adaptive rolling statistical baseline with Median Absolute Deviation (MAD)."""
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

@dataclass
class PhaseLifetimeProfile:
    """Tracks cumulative statistical profile across lifetime deployment using Welford's algorithm."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, val: float):
        """Numerically stable online update of running mean and variance."""
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.m2 += delta * delta2

    @property
    def std_dev(self) -> float:
        var = self.m2 / (self.count - 1) if self.count > 1 else 0.0016
        return float(np.sqrt(max(0.0016, var)))

    def to_dict(self) -> Dict:
        return {"count": self.count, "mean": round(self.mean, 4), "m2": round(self.m2, 4)}


def profile_from_dict(data: Dict) -> PhaseLifetimeProfile:
    """Deserializes lifetime profile dictionary."""
    return PhaseLifetimeProfile(count=data.get("count", 0), mean=data.get("mean", 0.0), m2=data.get("m2", 0.0))



@dataclass
class DetectionResult:
    """Outcome of a single observation evaluated against rolling MAD baseline."""
    # pylint: disable=too-many-instance-attributes
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
        # Lifelong machine statistical profiles (Welford's algorithm)
        self.lifetime_profiles: Dict[str, PhaseLifetimeProfile] = {}

    def seed_nominal_baselines(self, nominal_timings: Dict[str, float]):
        """Pre-seeds the buffer with nominal values to eliminate cold-start distortion."""
        for name, nominal in nominal_timings.items():
            self.history[name] = deque(
                [nominal] * (self.window_size // 2), maxlen=self.window_size
            )
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
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return 0.0, 0.0, 0.04
        median = float(np.median(valid))
        abs_deviations = np.abs(valid - median)
        mad = float(np.median(abs_deviations))

        sigma_robust = 1.4826 * mad
        std_val = float(np.std(valid)) if len(valid) > 1 else 0.04
        if not np.isfinite(sigma_robust) or sigma_robust < 0.04:
            sigma_robust = max(0.04, std_val if np.isfinite(std_val) else 0.04)

        return median, mad, sigma_robust

    def _classify_hypothesis(self, consec: int, excess_sec: float) -> str:
        """Determines the anomaly classification hypothesis based on duration and persistence."""
        if consec >= 4:
            return "CREEPING_WEAR"
        if excess_sec > 0.65:
            return "MICRO_STALL"
        return "VALVE_OVERLAP"

    def get_baseline_median(self, phase_name: str) -> float:
        """Returns current robust baseline median for phase, or 0.0 if uninitialized."""
        buf = self.history.get(phase_name)
        if buf and len(buf) > 0:
            valid = [x for x in buf if np.isfinite(x)]
            if valid:
                return float(np.median(valid))
        return 0.0

    def get_calibrated_sigma(self, phase_name: str, robust_sigma: float) -> float:
        """Blends short-term robust scale with lifetime empirical variance as deployment matures."""
        profile = self.lifetime_profiles.get(phase_name)
        if profile is None or profile.count < 30:
            return robust_sigma
        weight = min(0.65, profile.count / 400.0)
        return float((1.0 - weight) * robust_sigma + weight * profile.std_dev)

    def record_operator_feedback(
        self,
        phase_name: str,
        duration: float,
        is_fault: bool
    ):
        """Active learning: incorporates operator ground-truth to correct false alarms."""
        if not is_fault and self._is_valid_duration(duration):
            buffer = self._ensure_history_buffer(phase_name)
            buffer.append(duration)
            self.consecutive_anomalies[phase_name] = 0
            profile = self.lifetime_profiles.setdefault(phase_name, PhaseLifetimeProfile())
            profile.update(duration)

    def export_state(self) -> Dict:
        """Serializes learned baseline distribution buffers for deployment checkpointing."""
        return {
            "history": {k: list(v) for k, v in self.history.items()},
            "consecutive_anomalies": dict(self.consecutive_anomalies),
            "lifetime_profiles": {k: v.to_dict() for k, v in self.lifetime_profiles.items()},
            "window_size": self.window_size,
            "threshold_z": self.threshold_z
        }

    def import_state(self, state: Dict):
        """Restores learned baseline buffers from serialized deployment checkpoint."""
        for k, v in state.get("history", {}).items():
            self.history[k] = deque(v, maxlen=self.window_size)
        self.consecutive_anomalies = dict(state.get("consecutive_anomalies", {}))
        for k, v in state.get("lifetime_profiles", {}).items():
            self.lifetime_profiles[k] = profile_from_dict(v)

    def _is_valid_duration(self, duration: float) -> bool:
        """Validates that duration reading is finite and positive."""
        return isinstance(duration, (int, float)) and np.isfinite(duration) and duration > 0.0

    def _ensure_history_buffer(self, phase_name: str) -> deque:
        """Initializes phase buffer and anomaly counter if absent."""
        if phase_name not in self.history:
            self.history[phase_name] = deque(maxlen=self.window_size)
            self.consecutive_anomalies[phase_name] = 0
        return self.history[phase_name]

    def _update_normal_buffer(self, phase_name: str, duration: float) -> str:
        """Resets anomaly count and records duration into history and lifetime profile."""
        self.consecutive_anomalies[phase_name] = 0
        self.history[phase_name].append(duration)
        profile = self.lifetime_profiles.setdefault(phase_name, PhaseLifetimeProfile())
        profile.update(duration)
        return "NONE"

    def _update_anomaly_buffer(
        self,
        phase_name: str,
        duration: float,
        excess_sec: float
    ) -> str:
        """Increments anomaly counter and conditionally adapts to sustained shift."""
        self.consecutive_anomalies[phase_name] += 1
        consec = self.consecutive_anomalies[phase_name]
        hypothesis = self._classify_hypothesis(consec, excess_sec)
        if self.adapt_after_consecutive is not None and consec >= self.adapt_after_consecutive:
            self.history[phase_name].append(duration)
        return hypothesis

    def update_and_detect(
        self,
        cycle_id: int,
        phase_name: str,
        duration: float
    ) -> DetectionResult:
        """
        Ingests a new phase observation, evaluates against baseline,
        and conditionally updates the running buffer if within normal bounds.
        """
        if not self._is_valid_duration(duration):
            med = self.get_baseline_median(phase_name)
            return DetectionResult(
                cycle_id=cycle_id, phase_name=phase_name,
                actual_duration=0.0 if not np.isfinite(duration) else float(duration),
                baseline_median=round(med, 3), baseline_mad=0.0, robust_z_score=0.0,
                is_anomaly=False, excess_seconds=0.0, anomaly_type_hypothesis="INVALID_READING"
            )

        buffer = self._ensure_history_buffer(phase_name)
        if len(buffer) < self.min_samples_to_flag:
            buffer.append(duration)
            return DetectionResult(
                cycle_id=cycle_id, phase_name=phase_name, actual_duration=duration,
                baseline_median=duration, baseline_mad=0.0, robust_z_score=0.0,
                is_anomaly=False, excess_seconds=0.0, anomaly_type_hypothesis="WARMUP"
            )

        median, mad, sigma_robust = self._compute_robust_stats(buffer)
        calibrated_sigma = self.get_calibrated_sigma(phase_name, sigma_robust)
        z_score = (duration - median) / calibrated_sigma
        excess_sec = max(0.0, duration - (median + 1.645 * calibrated_sigma))
        is_anomaly = bool(z_score > self.threshold_z and excess_sec >= self.min_excess_sec)

        if is_anomaly:
            hypothesis = self._update_anomaly_buffer(phase_name, duration, excess_sec)
        else:
            hypothesis = self._update_normal_buffer(phase_name, duration)

        return DetectionResult(
            cycle_id=cycle_id, phase_name=phase_name,
            actual_duration=round(duration, 3), baseline_median=round(median, 3),
            baseline_mad=round(mad, 4), robust_z_score=round(z_score, 2),
            is_anomaly=is_anomaly, excess_seconds=round(excess_sec, 3),
            anomaly_type_hypothesis=hypothesis
        )


