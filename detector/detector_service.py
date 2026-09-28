"""Unified detection engine fusing Page's CUSUM with Tier 1, 2, and 3 anomaly analyzers."""
import csv
import logging
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from detector.changepoint_detector import ChangepointDetector
from detector.cusum_detector import CusumResult, PageCusumDetector
from detector.hmm_detector import HMMDetector
from detector.rolling_baseline import DetectionResult, RollingBaselineDetector
from simulator.press_config import PressConfig

logger = logging.getLogger("DetectorService")

TELEMETRY_COLUMNS = [
    "cycle_id",
    "phase_name",
    "actual_duration",
    "baseline_median",
    "robust_z_score",
    "is_anomaly",
    "excess_seconds",
    "anomaly_type",
    "hmm_state",
    "has_regime_drift",
    "tier1_anomaly",
    "cusum_alarm",
    "cusum_statistic",
]

CUSUM_WINDOW = 40
CUSUM_SEED_SAMPLES = 20
CUSUM_SIGMA_FLOOR_RATIO = 0.8


class DetectorService:
    """
    Fuses Page's CUSUM sequential test (primary), real-time robust statistical tracking
    (Tier 1), changepoint drift localization (Tier 2), and latent state decoding (Tier 3)
    across all press cycle phases.
    """

    def __init__(
        self,
        config: Optional[PressConfig] = None,
        telemetry_log_path: Optional[str] = None,
        threshold_z: float = 2.75,
        cusum_alpha: float = 0.01,
        cusum_beta: float = 0.05,
        append_to_log: bool = False
    ):
        """
        Args:
            telemetry_log_path: CSV file receiving one row per processed phase event.
                Defaults to data/telemetry_stream.csv.
            append_to_log: If False (default) the log is truncated so it only holds
                this session's events. Set True to resume an existing session's log.
        """
        self.config = config or PressConfig()
        self.tier1_baseline = RollingBaselineDetector(window_size=40, threshold_z=threshold_z)
        # Pre-seed baseline buffer with nominal design timings
        nominal_map = {name: t.nominal_sec for name, t in self.config.phases.items()}
        self.tier1_baseline.seed_nominal_baselines(nominal_map)

        # Primary detector: Page's CUSUM over its own outlier-quarantined rolling baseline
        self.cusum = PageCusumDetector(
            alpha=cusum_alpha,
            beta=cusum_beta,
            default_delta_sec=0.35,
            min_baseline_samples=15
        )
        self.cusum_histories: Dict[str, Deque[float]] = {}
        for name, timing in self.config.phases.items():
            self.cusum_histories[name] = deque([timing.nominal_sec] * CUSUM_SEED_SAMPLES, maxlen=CUSUM_WINDOW)

        self.tier2_changepoint = ChangepointDetector(penalty=3.0, model="rbf")

        # Per-phase duration history for changepoint and HMM analysis
        self.phase_histories: Dict[str, List[float]] = {
            name: [] for name in self.config.canonical_phase_names
        }
        self.active_drift_phases: set = set()

        # Tier 3 HMM detectors per dead-cycle phase initialized with domain priors
        self.tier3_hmms: Dict[str, HMMDetector] = {}
        for name, timing in self.config.phases.items():
            if timing.is_dead_cycle:
                detector = HMMDetector(n_states=3)
                detector.initialize_with_priors(timing.nominal_sec, timing.std_dev_sec)
                self.tier3_hmms[name] = detector

        # Telemetry output path
        if telemetry_log_path is None:
            self.telemetry_log_path = Path(__file__).resolve().parent.parent / "data" / "telemetry_stream.csv"
        else:
            self.telemetry_log_path = Path(telemetry_log_path)
        self.telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_csv(append_to_log)

    def _init_csv(self, append: bool):
        """Writes the telemetry header, truncating any previous session unless appending."""
        if append and self.telemetry_log_path.exists() and self.telemetry_log_path.stat().st_size > 0:
            return
        with open(self.telemetry_log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(TELEMETRY_COLUMNS)

    def rebaseline_phase(self, phase_name: str):
        """
        Forgets a phase's learned baseline, drift latch, and CUSUM evidence.
        Call after an acknowledged permanent change such as re-tooling or a repair.
        """
        self.tier1_baseline.rebaseline(phase_name)
        self.cusum_histories[phase_name] = deque(maxlen=CUSUM_WINDOW)
        self.cusum.reset_phase(phase_name)
        self.phase_histories[phase_name] = []
        self.active_drift_phases.discard(phase_name)
        logger.info("Baseline reset for phase %s", phase_name)

    def process_phase_event(
        self,
        cycle_id: int,
        phase_name: str,
        duration: float,
        is_dead_cycle: bool = True
    ) -> Dict:
        """
        Processes a single phase duration through the multi-tier detection pipeline.
        Returns a rich diagnostics dictionary.
        """
        history = self.phase_histories.setdefault(phase_name, [])
        history.append(duration)

        t1_result: DetectionResult = self.tier1_baseline.update_and_detect(
            cycle_id=cycle_id,
            phase_name=phase_name,
            duration=duration,
            is_dead_cycle=is_dead_cycle
        )
        cusum_result = self._evaluate_cusum(cycle_id, phase_name, duration)
        has_drift = self._update_drift_latch(phase_name, duration, t1_result)
        hmm_state = self._decode_hmm_state(phase_name)

        # Decision fusion:
        # 1. Page's CUSUM alarm (primary, SPRT-bounded error rates) OR
        # 2. Confirmed changepoint drift regime OR
        # 3. HMM predicts wear/stall state with mild excess duration
        is_anomaly = (
            cusum_result.is_alarm
            or has_drift
            or (hmm_state >= 1 and t1_result.excess_seconds > 0.20)
        )
        excess = self._excess_seconds(t1_result, cusum_result) if is_anomaly else 0.0

        record = {
            "cycle_id": cycle_id,
            "phase_name": phase_name,
            "actual_duration": round(duration, 3),
            "baseline_median": round(t1_result.baseline_median, 3),
            "robust_z_score": round(t1_result.robust_z_score, 2),
            "is_anomaly": is_anomaly,
            "excess_seconds": round(excess, 3),
            "anomaly_type": self._classify(is_anomaly, has_drift, excess),
            "hmm_state": int(hmm_state),
            "has_regime_drift": bool(has_drift),
            "tier1_anomaly": bool(t1_result.is_anomaly),
            "cusum_alarm": bool(cusum_result.is_alarm),
            "cusum_statistic": cusum_result.cumulative_sum,
        }

        self._log_record(record)
        return record

    def _evaluate_cusum(self, cycle_id: int, phase_name: str, duration: float) -> CusumResult:
        """Runs Page's CUSUM against the phase's quarantined median/MAD baseline."""
        hist = self.cusum_histories.setdefault(phase_name, deque(maxlen=CUSUM_WINDOW))
        mu, sigma = self._cusum_baseline(phase_name, hist, duration)

        result = self.cusum.evaluate_observation(
            cycle_id=cycle_id,
            phase_name=phase_name,
            duration=duration,
            baseline_mu=mu,
            baseline_sigma=sigma,
            sample_count=len(hist)
        )
        # Alarmed cycles are kept out of the baseline so a fault cannot mask itself
        if not result.is_alarm:
            hist.append(duration)
        return result

    def _cusum_baseline(self, phase_name: str, hist: Deque[float], duration: float) -> Tuple[float, float]:
        if not hist:
            return duration, 0.0
        arr = np.array(hist)
        mu = float(np.median(arr))
        mad = float(np.median(np.abs(arr - mu)))
        sigma = 1.4826 * mad
        timing = self.config.phases.get(phase_name)
        if timing is not None:
            sigma = max(sigma, timing.std_dev_sec * CUSUM_SIGMA_FLOOR_RATIO)
        return mu, sigma

    def _update_drift_latch(self, phase_name: str, duration: float, t1_result: DetectionResult) -> bool:
        """Latches a phase into CREEPING_WEAR once Pelt confirms an upward regime shift."""
        history = self.phase_histories[phase_name]
        consec = self.tier1_baseline.consecutive_anomalies.get(phase_name, 0)

        if len(history) >= 20 and consec >= 3:
            detected_cp_drift, _, _ = self.tier2_changepoint.evaluate_drift_regime(
                history,
                lookback_window=min(30, len(history))
            )
            if detected_cp_drift:
                self.active_drift_phases.add(phase_name)

        if phase_name not in self.active_drift_phases:
            return False
        # Release the latch once the duration returns to the normal baseline
        if duration > (t1_result.baseline_median + 0.15):
            return True
        self.active_drift_phases.discard(phase_name)
        return False

    def _decode_hmm_state(self, phase_name: str) -> int:
        if phase_name not in self.tier3_hmms:
            return 0
        return self.tier3_hmms[phase_name].predict_latest_state(
            self.phase_histories[phase_name],
            window_size=15
        )

    @staticmethod
    def _excess_seconds(t1_result: DetectionResult, cusum_result: CusumResult) -> float:
        # Tier 1 excess is measured above the 95% band, so it is the conservative
        # recoverable delay; fall back to the CUSUM residual for sequential alarms
        # that Tier 1 alone did not size.
        if t1_result.excess_seconds > 0.0:
            return t1_result.excess_seconds
        return cusum_result.excess_seconds

    @staticmethod
    def _classify(is_anomaly: bool, has_drift: bool, excess: float) -> str:
        if has_drift:
            return "CREEPING_WEAR"
        if not is_anomaly:
            return "NONE"
        return "MICRO_STALL" if excess > 0.60 else "VALVE_OVERLAP"

    def _log_record(self, r: Dict):
        """Append record to telemetry stream CSV."""
        with open(self.telemetry_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                r["cycle_id"],
                r["phase_name"],
                f"{r['actual_duration']:.3f}",
                f"{r['baseline_median']:.3f}",
                f"{r['robust_z_score']:.2f}",
                r["is_anomaly"],
                f"{r['excess_seconds']:.3f}",
                r["anomaly_type"],
                r["hmm_state"],
                r["has_regime_drift"],
                r["tier1_anomaly"],
                r["cusum_alarm"],
                f"{r['cusum_statistic']:.3f}",
            ])
