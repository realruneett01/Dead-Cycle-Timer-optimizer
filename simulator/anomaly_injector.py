"""Deterministic industrial anomaly injection engine for Extrusion Press simulation."""
import csv
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class AnomalyType(str, Enum):
    """Categorical classification of simulated hydraulic and mechanical faults."""
    NONE = "NONE"
    # Transient interlock/command overlap delay (+0.2s to +0.6s)
    VALVE_OVERLAP = "VALVE_OVERLAP"
    # Sudden spool stick-slip or proximity bounce (+0.4s to +1.2s)
    MICRO_STALL = "MICRO_STALL"
    # Progressive seal degradation or pilot clogging (+1.5%/cycle)
    CREEPING_WEAR = "CREEPING_WEAR"

@dataclass
class AnomalyEvent:
    """Ground truth audit event recorded for every injected physical anomaly."""
    cycle_id: int
    timestamp: str
    phase_name: str
    anomaly_type: AnomalyType
    injected_delay_sec: float
    nominal_duration: float
    actual_duration: float

class AnomalyInjector:
    """Injects calibrated failure modes and logs ground truth for benchmark validation."""

    def __init__(
        self,
        anomaly_probability: float = 0.12,
        ground_truth_path: Optional[str] = None,
        seed: Optional[int] = None
    ):
        self.anomaly_probability = anomaly_probability
        self.rng = random.Random(seed)

        # Ground truth file path
        if ground_truth_path is None:
            base_dir = Path(__file__).resolve().parent.parent / "data"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.ground_truth_path = base_dir / "ground_truth.csv"
        else:
            self.ground_truth_path = Path(ground_truth_path)
            self.ground_truth_path.parent.mkdir(parents=True, exist_ok=True)

        # Creeping wear active state: (phase_name, start_cycle, total_cycles, current_drift_pct)
        self.active_wear_drift: Optional[Dict] = None
        self._init_csv()

    def _init_csv(self):
        """Initialize ground truth CSV file with headers if missing."""
        if not self.ground_truth_path.exists() or self.ground_truth_path.stat().st_size == 0:
            with open(self.ground_truth_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "cycle_id",
                    "timestamp",
                    "phase_name",
                    "anomaly_type",
                    "injected_delay_sec",
                    "nominal_duration",
                    "actual_duration"
                ])

    def _evaluate_active_wear(
        self, cycle_id: int
    ) -> Optional[Tuple[str, Tuple[AnomalyType, float]]]:
        """Calculates delay if a creeping wear progression is active, or clears it when finished."""
        if self.active_wear_drift is None:
            return None

        phase = self.active_wear_drift["phase"]
        step = cycle_id - self.active_wear_drift["start_cycle"]
        total_steps = self.active_wear_drift["total_cycles"]

        if not 0 <= step < total_steps:
            self.active_wear_drift = None
            return None

        base_pct = self.active_wear_drift["base_pct"]
        rate = self.active_wear_drift["rate_per_cycle"]
        cumulative_pct = base_pct + (step * rate)
        nominal = self.active_wear_drift["nominal_sec"]
        delay = round(nominal * cumulative_pct, 3)
        return phase, (AnomalyType.CREEPING_WEAR, delay)

    def _create_creeping_wear(
        self, cycle_id: int
    ) -> Optional[Tuple[str, Tuple[AnomalyType, float]]]:
        """Initializes a new creeping wear progression."""
        if self.active_wear_drift is not None:
            return None
        target_phase = self.rng.choice(["decompression", "container_shift_close"])
        total_cycles = self.rng.randint(25, 40)
        base_pct = 0.20
        rate = (0.38 - 0.20) / total_cycles
        nominal_sec = 2.2 if target_phase == "decompression" else 2.8
        self.active_wear_drift = {
            "phase": target_phase,
            "start_cycle": cycle_id,
            "total_cycles": total_cycles,
            "base_pct": base_pct,
            "rate_per_cycle": rate,
            "nominal_sec": nominal_sec
        }
        delay = round(nominal_sec * base_pct, 3)
        return target_phase, (AnomalyType.CREEPING_WEAR, delay)

    def _sample_new_anomaly(self, cycle_id: int) -> Optional[Tuple[str, Tuple[AnomalyType, float]]]:
        """Stochastically samples and generates a new anomaly event."""
        roll = self.rng.random()
        if roll >= self.anomaly_probability:
            return None

        choice = self.rng.choices(
            [AnomalyType.VALVE_OVERLAP, AnomalyType.MICRO_STALL, AnomalyType.CREEPING_WEAR],
            weights=[0.45, 0.40, 0.15],
            k=1
        )[0]

        if choice == AnomalyType.VALVE_OVERLAP:
            target_phase = self.rng.choice(
                ["container_shift_open", "shear_stroke", "container_shift_close"]
            )
            delay = round(self.rng.uniform(0.35, 0.70), 3)
            return target_phase, (AnomalyType.VALVE_OVERLAP, delay)

        if choice == AnomalyType.MICRO_STALL:
            target_phase = self.rng.choice(["billet_load", "shear_stroke"])
            delay = round(self.rng.uniform(0.50, 1.25), 3)
            return target_phase, (AnomalyType.MICRO_STALL, delay)

        return self._create_creeping_wear(cycle_id)

    def evaluate_cycle_anomalies(
        self,
        cycle_id: int,
        phases: Optional[List[str]] = None
    ) -> Dict[str, Tuple[AnomalyType, float]]:
        # pylint: disable=unused-argument
        """
        Determines if any anomaly should be injected for each phase of a given cycle.
        Returns a dict: {phase_name: (AnomalyType, injected_delay_sec)}
        """
        results: Dict[str, Tuple[AnomalyType, float]] = {}

        wear_event = self._evaluate_active_wear(cycle_id)
        if wear_event is not None:
            results[wear_event[0]] = wear_event[1]

        new_event = self._sample_new_anomaly(cycle_id)
        if new_event is not None:
            results[new_event[0]] = new_event[1]

        return results

    def log_anomaly(self, event: AnomalyEvent):
        """Append ground truth event to CSV."""
        with open(self.ground_truth_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                event.cycle_id,
                event.timestamp,
                event.phase_name,
                event.anomaly_type.value,
                f"{event.injected_delay_sec:.3f}",
                f"{event.nominal_duration:.3f}",
                f"{event.actual_duration:.3f}"
            ])
