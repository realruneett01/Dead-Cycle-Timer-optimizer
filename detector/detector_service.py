"""Unified detection engine coordinating Tier 1, 2, and 3 anomaly analyzers."""
import csv
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from detector.changepoint_detector import ChangepointDetector
from detector.hmm_detector import HMMDetector
from detector.rolling_baseline import DetectionResult, RollingBaselineDetector
from simulator.press_config import PressConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] DetectorService: %(message)s")
logger = logging.getLogger("DetectorService")

class DetectorService:
    """
    Coordinates real-time statistical tracking (Tier 1), changepoint drift localization (Tier 2),
    and latent state decoding (Tier 3) across all press cycle phases.
    """

    def __init__(
        self,
        config: Optional[PressConfig] = None,
        telemetry_log_path: Optional[str] = None,
        threshold_z: float = 2.75
    ):
        self.config = config or PressConfig()
        self.tier1_baseline = RollingBaselineDetector(window_size=40, threshold_z=threshold_z)
        # Pre-seed baseline buffer with nominal design timings
        nominal_map = {name: t.nominal_sec for name, t in self.config.phases.items()}
        self.tier1_baseline.seed_nominal_baselines(nominal_map)
        
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
            base_dir = Path(__file__).resolve().parent.parent / "data"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.telemetry_log_path = base_dir / "telemetry_stream.csv"
        else:
            self.telemetry_log_path = Path(telemetry_log_path)
            self.telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)
            
        self._init_csv()

    def _init_csv(self):
        """Initializes telemetry log file with headers."""
        if not self.telemetry_log_path.exists() or self.telemetry_log_path.stat().st_size == 0:
            with open(self.telemetry_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "cycle_id",
                    "phase_name",
                    "actual_duration",
                    "baseline_median",
                    "robust_z_score",
                    "is_anomaly",
                    "excess_seconds",
                    "anomaly_type",
                    "hmm_state",
                    "has_regime_drift"
                ])

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
        # Append to phase history
        if phase_name in self.phase_histories:
            self.phase_histories[phase_name].append(duration)
        else:
            self.phase_histories[phase_name] = [duration]

        # 1. Tier 1: Robust Rolling Baseline (Z-score + MAD)
        t1_result: DetectionResult = self.tier1_baseline.update_and_detect(
            cycle_id=cycle_id,
            phase_name=phase_name,
            duration=duration,
            is_dead_cycle=is_dead_cycle
        )

        # 2. Tier 2: Ruptures Changepoint Detection (Evaluated periodically or when consecutively flagged)
        has_drift = False
        consec = self.tier1_baseline.consecutive_anomalies.get(phase_name, 0)
        
        if len(self.phase_histories[phase_name]) >= 20 and consec >= 3:
            detected_cp_drift, cp_idx, delta = self.tier2_changepoint.evaluate_drift_regime(
                self.phase_histories[phase_name],
                lookback_window=min(30, len(self.phase_histories[phase_name]))
            )
            if detected_cp_drift:
                self.active_drift_phases.add(phase_name)

        if phase_name in self.active_drift_phases:
            # Check if duration is still elevated above normal baseline
            if duration > (t1_result.baseline_median + 0.15):
                has_drift = True
            else:
                self.active_drift_phases.discard(phase_name)

        # 3. Tier 3: Gaussian HMM Latent State Decoding
        hmm_state = 0
        if phase_name in self.tier3_hmms:
            hmm_state = self.tier3_hmms[phase_name].predict_latest_state(
                self.phase_histories[phase_name],
                window_size=15
            )

        # Composite decision logic:
        # 1. Tier 1 flag OR
        # 2. Confirmed changepoint drift regime OR
        # 3. HMM predicts wear/stall state with mild excess duration
        is_anomaly = (
            t1_result.is_anomaly
            or has_drift
            or (hmm_state >= 1 and t1_result.excess_seconds > 0.20)
        )
        
        # Refine anomaly classification
        if has_drift:
            final_anomaly_type = "CREEPING_WEAR"
        elif is_anomaly:
            if t1_result.excess_seconds > 0.60:
                final_anomaly_type = "MICRO_STALL"
            else:
                final_anomaly_type = "VALVE_OVERLAP"
        else:
            final_anomaly_type = "NONE"

        record = {
            "cycle_id": cycle_id,
            "phase_name": phase_name,
            "actual_duration": round(duration, 3),
            "baseline_median": round(t1_result.baseline_median, 3),
            "robust_z_score": round(t1_result.robust_z_score, 2),
            "is_anomaly": is_anomaly,
            "excess_seconds": round(t1_result.excess_seconds, 3) if is_anomaly else 0.0,
            "anomaly_type": final_anomaly_type,
            "hmm_state": int(hmm_state),
            "has_regime_drift": bool(has_drift)
        }

        # Log to telemetry CSV
        self._log_record(record)
        return record

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
                r["has_regime_drift"]
            ])
