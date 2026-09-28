"""Unified detection engine fusing Page's CUSUM with Tier 1, 2, and 3 anomaly analyzers."""
import csv
import json
import logging
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, TextIO, Tuple

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


@dataclass
class DetectorServiceConfig:
    """Configuration options for DetectorService."""
    config: Optional[PressConfig] = None
    telemetry_log_path: Optional[str] = None
    threshold_z: float = 2.75
    cusum_alpha: float = 0.01
    cusum_beta: float = 0.05
    append_to_log: bool = False


@dataclass
class PhaseEventObservation:
    """Encapsulates a single phase event observation."""
    cycle_id: int
    phase_name: str
    duration: float
    is_dead_cycle: bool = True


class DetectorService:
    """
    Fuses Page's CUSUM sequential test (primary), real-time robust statistical tracking
    (Tier 1), changepoint drift localization (Tier 2), and latent state decoding (Tier 3)
    across all press cycle phases.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        service_config: Optional[DetectorServiceConfig] = None,
        **kwargs
    ):
        """
        Initializes multi-tier anomaly detection engines and telemetry logger.
        """
        cfg = service_config or DetectorServiceConfig(
            config=kwargs.get("config"),
            telemetry_log_path=kwargs.get("telemetry_log_path"),
            threshold_z=kwargs.get("threshold_z", 2.75),
            cusum_alpha=kwargs.get("cusum_alpha", 0.01),
            cusum_beta=kwargs.get("cusum_beta", 0.05),
            append_to_log=kwargs.get("append_to_log", False)
        )
        self.config = cfg.config or PressConfig()
        self.tier1_baseline = RollingBaselineDetector(window_size=40, threshold_z=cfg.threshold_z)
        # Pre-seed baseline buffer with nominal design timings
        nominal_map = {name: t.nominal_sec for name, t in self.config.phases.items()}
        self.tier1_baseline.seed_nominal_baselines(nominal_map)

        # Primary detector: Page's CUSUM over its own outlier-quarantined rolling baseline
        self.cusum = PageCusumDetector(
            alpha=cfg.cusum_alpha,
            beta=cfg.cusum_beta,
            default_delta_sec=0.35,
            min_baseline_samples=15
        )
        self.cusum_histories: Dict[str, Deque[float]] = {}
        for name, timing in self.config.phases.items():
            self.cusum_histories[name] = deque(
                [timing.nominal_sec] * CUSUM_SEED_SAMPLES, maxlen=CUSUM_WINDOW
            )

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
        if cfg.telemetry_log_path is None:
            base_data = Path(__file__).resolve().parent.parent / "data"
            self.telemetry_log_path = base_data / "telemetry_stream.csv"
        else:
            self.telemetry_log_path = Path(cfg.telemetry_log_path)
        self.telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)

        self._log_file: Optional[TextIO] = None
        self._csv_writer = None
        self._init_csv(cfg.append_to_log)

    def _init_csv(self, append: bool):
        """Initializes telemetry CSV and establishes an open writer handle."""
        write_header = not (
            append
            and self.telemetry_log_path.exists()
            and self.telemetry_log_path.stat().st_size > 0
        )
        mode = "a" if append else "w"
        self._log_file = open(self.telemetry_log_path, mode, newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._log_file)
        if write_header:
            self._csv_writer.writerow(TELEMETRY_COLUMNS)
            self._log_file.flush()

    def close(self):
        """Closes telemetry log file handle if open."""
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.flush()
            self._log_file.close()
        self._csv_writer = None

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

    def adapt_to_new_regime(self, phase_name: str, observations: List[float]):
        """
        Online learning: rapidly adapts baseline and models to a new operating regime (e.g. after re-tooling).
        """
        valid = [float(x) for x in observations if isinstance(x, (int, float)) and np.isfinite(x) and x > 0.0]
        if not valid:
            return
        self.tier1_baseline.rebaseline(phase_name)
        for val in valid:
            self.tier1_baseline.history[phase_name].append(val)
        self.cusum_histories[phase_name] = deque(valid[-CUSUM_WINDOW:], maxlen=CUSUM_WINDOW)
        self.cusum.reset_phase(phase_name)
        self.phase_histories[phase_name] = list(valid)
        self.active_drift_phases.discard(phase_name)
        if phase_name in self.tier3_hmms and len(valid) >= 30:
            self.tier3_hmms[phase_name].adapt_to_stream(valid)
        logger.info("Adapted phase %s to new regime with %d samples", phase_name, len(valid))

    def save_model_checkpoint(self, checkpoint_path: str):
        """Serializes complete multi-tier detector state into a JSON checkpoint for edge deployment."""
        state = {
            "tier1_baseline": self.tier1_baseline.export_state(),
            "cusum": self.cusum.export_state(),
            "cusum_histories": {k: list(v) for k, v in self.cusum_histories.items()},
            "active_drift_phases": list(self.active_drift_phases),
            "tier3_hmms": {k: v.export_state() for k, v in self.tier3_hmms.items()}
        }
        dest = Path(checkpoint_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        logger.info("Saved detector checkpoint to %s", checkpoint_path)

    def _restore_checkpoint_baselines(self, state: Dict):
        """Restores Tier 1 and CUSUM running state from checkpoint dictionary."""
        if "tier1_baseline" in state:
            self.tier1_baseline.import_state(state["tier1_baseline"])
        if "cusum" in state:
            self.cusum.import_state(state["cusum"])
        for k, v in state.get("cusum_histories", {}).items():
            self.cusum_histories[k] = deque(v, maxlen=CUSUM_WINDOW)
        if "active_drift_phases" in state:
            self.active_drift_phases = set(state["active_drift_phases"])

    def _restore_checkpoint_hmms(self, state: Dict):
        """Restores Tier 3 HMM parameters from checkpoint dictionary."""
        for k, v in state.get("tier3_hmms", {}).items():
            if k in self.tier3_hmms:
                self.tier3_hmms[k].import_state(v)

    def load_model_checkpoint(self, checkpoint_path: str):
        """Restores complete multi-tier detector state from an edge deployment JSON checkpoint."""
        src = Path(checkpoint_path)
        if not src.exists():
            logger.warning("Checkpoint %s does not exist; skipping load.", checkpoint_path)
            return
        with open(src, "r", encoding="utf-8") as f:
            state = json.load(f)
        self._restore_checkpoint_baselines(state)
        self._restore_checkpoint_hmms(state)
        logger.info("Loaded detector checkpoint from %s", checkpoint_path)

    def record_operator_feedback(
        self,
        phase_name: str,
        duration: float,
        is_fault: bool
    ):
        """
        Incorporates human-in-the-loop operator feedback for lifelong model adaptation.
        If is_fault=False (operator dismissed false alarm), updates baselines and histories.
        """
        self.tier1_baseline.record_operator_feedback(phase_name, duration, is_fault)
        if not is_fault and phase_name in self.cusum_histories:
            self.cusum_histories[phase_name].append(duration)


    @staticmethod
    def _extract_event_tuple(
        event: Optional[PhaseEventObservation],
        kwargs: Dict
    ) -> Tuple[int, str, float]:
        """Extracts and sanitizes cycle_id, phase_name, and duration from inputs."""
        if event is not None:
            raw = event.duration
            dur = float(raw) if (isinstance(raw, (int, float)) and np.isfinite(raw) and raw > 0.0) else 0.0
            return event.cycle_id, event.phase_name, dur
        raw = kwargs["duration"]
        dur = float(raw) if (isinstance(raw, (int, float)) and np.isfinite(raw) and raw > 0.0) else 0.0
        return kwargs["cycle_id"], kwargs["phase_name"], dur

    @staticmethod
    def _fuse_anomaly_decision(
        cusum_alarm: bool,
        has_drift: bool,
        hmm_state: int,
        t1_excess: float
    ) -> bool:
        """Evaluates fused multi-tier criteria across sequential SPRT, Pelt drift, and HMM."""
        return cusum_alarm or has_drift or (hmm_state >= 1 and t1_excess > 0.20)

    def process_phase_event(
        self,
        event: Optional[PhaseEventObservation] = None,
        **kwargs
    ) -> Dict:
        """
        Processes a single phase duration through the multi-tier detection pipeline.
        Returns a rich diagnostics dictionary.
        """
        cycle_id, phase_name, duration = self._extract_event_tuple(event, kwargs)

        history = self.phase_histories.setdefault(phase_name, [])
        history.append(duration)
        if len(history) % 100 == 0 and phase_name in self.tier3_hmms:
            self.tier3_hmms[phase_name].adapt_to_stream(history[-120:])

        t1_result: DetectionResult = self.tier1_baseline.update_and_detect(
            cycle_id=cycle_id,
            phase_name=phase_name,
            duration=duration
        )
        cusum_result = self._evaluate_cusum(cycle_id, phase_name, duration)
        has_drift = self._update_drift_latch(phase_name, duration, t1_result)
        hmm_state = self._decode_hmm_state(phase_name)

        is_anomaly = self._fuse_anomaly_decision(
            cusum_result.is_alarm, has_drift, hmm_state, t1_result.excess_seconds
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

    def _cusum_baseline(
        self, phase_name: str, hist: Deque[float], duration: float
    ) -> Tuple[float, float]:
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

    def _update_drift_latch(
        self, phase_name: str, duration: float, t1_result: DetectionResult
    ) -> bool:
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
        """Append record to telemetry stream CSV using persistent writer."""
        if self._csv_writer is None or self._log_file is None or self._log_file.closed:
            return
        self._csv_writer.writerow([
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
        self._log_file.flush()
