"""Per-machine press profiles: JSON overrides on PressConfig and calibration from observed data."""
import csv
import json
import logging
from dataclasses import fields, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from simulator.press_config import PhaseTiming, PressConfig

logger = logging.getLogger("PressProfile")

MIN_CALIBRATION_SAMPLES = 30
DURATION_COLUMNS = ("actual_duration", "duration", "duration_sec")
PHASE_BOUND_SIGMAS = 6.0
MAD_TO_SIGMA = 1.4826


def load_press_config(path: Optional[str]) -> PressConfig:
    """Builds a PressConfig from a profile JSON; an empty path yields the built-in 28 MN defaults.

    Existing phases are patched field-by-field. New phases must define every PhaseTiming field.
    """
    config = PressConfig()
    if not path:
        return config
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("press_id", "rated_tonnage_mn", "nominal_billet_weight_kg",
                "nominal_billet_length_mm", "nominal_billet_diameter_mm"):
        if key in data:
            setattr(config, key, data[key])
    if data.get("replace_phases"):
        config.phases = {}
    allowed = {f.name for f in fields(PhaseTiming)}
    for name, spec in data.get("phases", {}).items():
        bad = sorted(set(spec) - allowed)
        if bad:
            raise ValueError(f"Phase '{name}': unknown field(s) {', '.join(bad)}")
        if name in config.phases:
            config.phases[name] = replace(config.phases[name], **spec)
        else:
            config.phases[name] = PhaseTiming(name=name, **spec)
    return config


def _read_durations(csv_path: Path) -> Dict[str, List[float]]:
    samples: Dict[str, List[float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = next((c for c in DURATION_COLUMNS if reader.fieldnames and c in reader.fieldnames), None)
        if col is None or "phase_name" not in (reader.fieldnames or []):
            raise ValueError(
                f"{csv_path} needs a 'phase_name' column and one of {DURATION_COLUMNS}"
            )
        for row in reader:
            try:
                value = float(row[col])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0.0:
                samples.setdefault(row["phase_name"], []).append(value)
    return samples


def calibrate_from_csv(csv_path: str, base: Optional[PressConfig] = None) -> Dict:
    """Learns per-phase nominal/sigma/bounds from a healthy-operation log.

    Median and MAD are used so a handful of faults in the log do not skew the baseline.
    Returns a profile dict suitable for json.dump and load_press_config.
    """
    base = base or PressConfig()
    samples = _read_durations(Path(csv_path))
    phases: Dict[str, Dict] = {}
    for name, values in samples.items():
        if name not in base.phases:
            logger.warning("Ignoring unknown phase '%s' (%d samples)", name, len(values))
            continue
        if len(values) < MIN_CALIBRATION_SAMPLES:
            logger.warning("Phase '%s' has %d samples (<%d); keeping defaults",
                           name, len(values), MIN_CALIBRATION_SAMPLES)
            continue
        arr = np.array(values)
        median = float(np.median(arr))
        sigma = float(MAD_TO_SIGMA * np.median(np.abs(arr - median)))
        sigma = max(sigma, 0.01)
        phases[name] = {
            "nominal_sec": round(median, 3),
            "std_dev_sec": round(sigma, 3),
            "min_sec": round(max(0.1, median - PHASE_BOUND_SIGMAS * sigma), 3),
            "max_sec": round(median + PHASE_BOUND_SIGMAS * sigma * 2, 3),
        }
    return {"press_id": base.press_id, "phases": phases}
