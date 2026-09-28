"""Benchmark validation script evaluating detector Precision,
Recall, F1, and FPR against Ground Truth.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=wrong-import-position
import pandas as pd
from detector.detector_service import DetectorService
from simulator.anomaly_injector import AnomalyInjector
from simulator.press_config import PressConfig
from simulator.press_state_machine import PressStateMachine

ANOMALY_TYPES = ["VALVE_OVERLAP", "MICRO_STALL", "CREEPING_WEAR"]

# Telemetry column holding each detector's verdict -> (results key, report label)
DETECTORS = {
    "tier1_anomaly": ("baseline_detector", "Rolling MAD Baseline"),
    "cusum_alarm": ("page_cusum_detector", "Page CUSUM (SPRT)"),
    "is_anomaly": ("fused_detector_service", "Fused Service"),
}

Event = Tuple[int, str]


@dataclass
class BenchmarkReportSummary:
    """Holds summary counts for tabular benchmark reporting."""
    num_cycles: int
    warmup_cutoff: int
    total_events: int
    gt_count: int


@dataclass
class BenchmarkConfig:
    """Execution parameters for detector benchmark validation."""
    num_cycles: int = 600
    anomaly_prob: float = 0.12
    seed: int = 42
    threshold_z: float = 2.75
    cusum_alpha: float = 0.01
    cusum_beta: float = 0.05
    output_dir: Path = PROJECT_ROOT / "data"


def score_detector(
    detected: Set[Event],
    gt_set: Set[Event],
    gt_type_map: Dict[Event, str],
    total_events: int
) -> dict:
    # pylint: disable=too-many-locals
    """Computes confusion matrix, headline metrics, and per-class recall for one detector."""
    tp = len(detected & gt_set)
    fp = len(detected - gt_set)
    fn = len(gt_set - detected)
    tn = total_events - (tp + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    class_recall = {}
    for anom_type in ANOMALY_TYPES:
        type_gt = {k for k, v in gt_type_map.items() if v == anom_type}
        hits = len(type_gt & detected)
        class_recall[anom_type] = {
            "total": len(type_gt),
            "detected": hits,
            "recall": round(hits / len(type_gt), 4) if type_gt else 0.0,
        }
    # Unweighted mean across fault classes, so the abundant CREEPING_WEAR
    # events cannot mask weak detection of the rarer transient faults
    macro_recall = sum(c["recall"] for c in class_recall.values()) / len(ANOMALY_TYPES)

    return {
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "macro_recall": round(macro_recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "class_recall": class_recall,
    }


def print_report(results: dict, summary: BenchmarkReportSummary) -> None:
    """Formats and prints comparative evaluation table across evaluated detectors."""
    scores = [results[key] for key, _ in DETECTORS.values()]
    labels = [label for _, label in DETECTORS.values()]
    width = 24 + 3 * 23

    def row(name, values):
        print(f"{name:<24}" + "".join(f" | {v:<20}" for v in values))

    print("=" * width)
    print("       DCTO ANOMALY DETECTOR BENCHMARK: BASELINE vs. PAGE'S CUSUM vs. FUSED")
    print("=" * width)
    print(
        f"Evaluated Cycles:      {summary.num_cycles - summary.warmup_cutoff} "
        f"(warmup: {summary.warmup_cutoff} excluded)"
    )
    print(f"Evaluated Events:      {summary.total_events}")
    print(f"Ground Truth Anomalies:{summary.gt_count}")
    print("-" * width)
    row("Metric", labels)
    print("-" * width)
    for key, name in [("TP", "True Positives (TP)"), ("FP", "False Positives (FP)"),
                      ("FN", "False Negatives (FN)"), ("TN", "True Negatives (TN)")]:
        row(name, [s["confusion_matrix"][key] for s in scores])
    print("-" * width)
    row("Precision", [f"{s['precision'] * 100:.2f}%" for s in scores])
    row("Recall (Sensitivity)", [f"{s['recall'] * 100:.2f}%" for s in scores])
    row("Macro Recall (by class)", [f"{s['macro_recall'] * 100:.2f}%" for s in scores])
    row("F1-Score", [f"{s['f1_score']:.4f}" for s in scores])
    row("False Positive Rate", [f"{s['false_positive_rate'] * 100:.2f}%" for s in scores])
    print("-" * width)
    print("Per-Class Recall Breakdown:")
    for c_name in ANOMALY_TYPES:
        cells = [
            f"{s['class_recall'][c_name]['recall'] * 100:.1f}% "
            f"({s['class_recall'][c_name]['detected']}/{s['class_recall'][c_name]['total']})"
            for s in scores
        ]
        row(f"  - {c_name}", cells)
    print("=" * width)


def _setup_benchmark_environment(
    config: BenchmarkConfig
) -> Tuple[Path, Path, Path, PressStateMachine, DetectorService]:
    """Initializes paths, clears stale artifacts, and sets up state machine and detector."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_file = out_dir / "ground_truth.csv"
    telem_file = out_dir / "telemetry_stream.csv"
    metrics_file = out_dir / "validation_results.json"

    if gt_file.exists():
        gt_file.unlink()

    press_cfg = PressConfig()
    injector = AnomalyInjector(
        anomaly_probability=config.anomaly_prob,
        ground_truth_path=str(gt_file),
        seed=config.seed
    )
    sm = PressStateMachine(config=press_cfg, injector=injector, seed=config.seed)
    detector = DetectorService(
        config=press_cfg,
        telemetry_log_path=str(telem_file),
        threshold_z=config.threshold_z,
        cusum_alpha=config.cusum_alpha,
        cusum_beta=config.cusum_beta
    )
    return gt_file, telem_file, metrics_file, sm, detector


def _simulate_cycles(
    sm: PressStateMachine,
    detector: DetectorService,
    num_cycles: int
) -> None:
    """Executes state machine cycles and passes events to detector service."""
    for cycle_id in range(1, num_cycles + 1):
        for ev in sm.run_cycle(cycle_id=cycle_id):
            detector.process_phase_event(
                cycle_id=ev.cycle_id,
                phase_name=ev.phase_name,
                duration=ev.duration_sec,
                is_dead_cycle=ev.is_dead_cycle
            )


def _evaluate_benchmark(
    gt_file: Path,
    telem_file: Path,
    warmup_cutoff: int,
    config: BenchmarkConfig
) -> Tuple[dict, BenchmarkReportSummary]:
    # pylint: disable=too-many-locals
    """Computes benchmark metrics from generated telemetry and ground truth logs."""
    df_gt = pd.read_csv(gt_file)
    df_telem = pd.read_csv(telem_file)
    df_gt_eval = df_gt[df_gt["cycle_id"] > warmup_cutoff]
    df_telem_eval = df_telem[df_telem["cycle_id"] > warmup_cutoff]

    gt_keys = list(zip(df_gt_eval["cycle_id"], df_gt_eval["phase_name"]))
    gt_set = set(gt_keys)
    gt_type_map = dict(zip(gt_keys, df_gt_eval["anomaly_type"]))
    total_events = len(set(zip(df_telem_eval["cycle_id"], df_telem_eval["phase_name"])))

    results = {
        "benchmark_parameters": {
            "num_cycles": config.num_cycles,
            "warmup_cycles_excluded": warmup_cutoff,
            "anomaly_probability": config.anomaly_prob,
            "random_seed": config.seed,
            "threshold_z": config.threshold_z,
            "cusum_alpha": config.cusum_alpha,
            "cusum_beta": config.cusum_beta
        }
    }
    for column, (key, _) in DETECTORS.items():
        flagged = df_telem_eval[df_telem_eval[column].astype(bool)]
        detected = set(zip(flagged["cycle_id"], flagged["phase_name"]))
        results[key] = score_detector(detected, gt_set, gt_type_map, total_events)

    summary = BenchmarkReportSummary(
        num_cycles=config.num_cycles,
        warmup_cutoff=warmup_cutoff,
        total_events=total_events,
        gt_count=len(gt_set)
    )
    return results, summary


def run_benchmark(config: Optional[BenchmarkConfig] = None, **kwargs) -> dict:
    """
    Executes a benchmark simulation run through DetectorService and scores each of its
    verdicts (Tier 1 rolling MAD baseline, Page's CUSUM, and the fused decision)
    against ground-truth injected anomalies.
    """
    if config is None:
        config = BenchmarkConfig(**kwargs) if kwargs else BenchmarkConfig()
    elif kwargs:
        for key, value in kwargs.items():
            setattr(config, key, value)

    gt_file, telem_file, metrics_file, sm, detector = _setup_benchmark_environment(config)

    print(
        f"\n--- Running Benchmark Simulation ({config.num_cycles} cycles, "
        f"p={config.anomaly_prob}, threshold_z={config.threshold_z}) ---"
    )

    _simulate_cycles(sm, detector, config.num_cycles)

    warmup_cutoff = 25
    results, summary = _evaluate_benchmark(gt_file, telem_file, warmup_cutoff, config)

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_report(results, summary)
    print(f"Metrics saved to: {metrics_file}\n")

    return results


if __name__ == "__main__":
    run_benchmark()
