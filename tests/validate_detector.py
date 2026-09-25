"""Benchmark validation script evaluating detector Precision, Recall, F1, and FPR against Ground Truth."""
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from detector.detector_service import DetectorService
from simulator.anomaly_injector import AnomalyInjector
from simulator.press_config import PressConfig
from simulator.press_state_machine import PressStateMachine

def run_benchmark(
    num_cycles: int = 600,
    anomaly_prob: float = 0.12,
    seed: int = 42,
    threshold_z: float = 2.75
) -> dict:
    """
    Executes a benchmark simulation run, streams events through DetectorService,
    and compares detections against ground-truth injected anomalies.
    """
    base_dir = Path(__file__).resolve().parent.parent / "data"
    base_dir.mkdir(parents=True, exist_ok=True)
    gt_file = base_dir / "ground_truth.csv"
    telem_file = base_dir / "telemetry_stream.csv"
    metrics_file = base_dir / "validation_results.json"

    # Clean previous benchmark artifacts
    if gt_file.exists():
        gt_file.unlink()
    if telem_file.exists():
        telem_file.unlink()

    config = PressConfig()
    injector = AnomalyInjector(anomaly_probability=anomaly_prob, ground_truth_path=str(gt_file), seed=seed)
    sm = PressStateMachine(config=config, injector=injector, seed=seed)
    detector = DetectorService(config=config, telemetry_log_path=str(telem_file), threshold_z=threshold_z)

    print(f"\n--- Running Benchmark Simulation ({num_cycles} cycles, p={anomaly_prob}, threshold_z={threshold_z}) ---")
    
    # Process all cycles
    for cycle_id in range(1, num_cycles + 1):
        events = sm.run_cycle(cycle_id=cycle_id)
        for ev in events:
            detector.process_phase_event(
                cycle_id=ev.cycle_id,
                phase_name=ev.phase_name,
                duration=ev.duration_sec,
                is_dead_cycle=ev.is_dead_cycle
            )

    # Load produced datasets
    df_gt = pd.read_csv(gt_file)
    df_telem = pd.read_csv(telem_file)

    # Exclude initial warmup period (first 25 cycles) for fair evaluation
    warmup_cutoff = 25
    df_gt_eval = df_gt[df_gt["cycle_id"] > warmup_cutoff].copy()
    df_telem_eval = df_telem[df_telem["cycle_id"] > warmup_cutoff].copy()

    # Create ground truth set of (cycle_id, phase_name)
    gt_set = set(zip(df_gt_eval["cycle_id"], df_gt_eval["phase_name"]))
    gt_type_map = dict(zip(zip(df_gt_eval["cycle_id"], df_gt_eval["phase_name"]), df_gt_eval["anomaly_type"]))

    # Identified anomalies by detector
    detected_mask = df_telem_eval["is_anomaly"] == True
    detected_set = set(zip(df_telem_eval.loc[detected_mask, "cycle_id"], df_telem_eval.loc[detected_mask, "phase_name"]))

    # Total evaluated events
    all_events_set = set(zip(df_telem_eval["cycle_id"], df_telem_eval["phase_name"]))
    total_eval_events = len(all_events_set)

    # Classification counts
    tp = len(detected_set.intersection(gt_set))
    fp = len(detected_set - gt_set)
    fn = len(gt_set - detected_set)
    tn = total_eval_events - (tp + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Per-anomaly-type breakdown
    breakdown = {}
    for anom_type in ["VALVE_OVERLAP", "MICRO_STALL", "CREEPING_WEAR"]:
        type_gt = {k for k, v in gt_type_map.items() if v == anom_type}
        type_tp = len(type_gt.intersection(detected_set))
        type_total = len(type_gt)
        type_recall = type_tp / type_total if type_total > 0 else 0.0
        breakdown[anom_type] = {
            "ground_truth_count": type_total,
            "detected_count": type_tp,
            "recall": round(type_recall, 4)
        }

    results = {
        "benchmark_parameters": {
            "num_cycles": num_cycles,
            "warmup_cycles_excluded": warmup_cutoff,
            "anomaly_probability": anomaly_prob,
            "random_seed": seed,
            "threshold_z": threshold_z
        },
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total_evaluated_events": total_eval_events
        },
        "performance_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4)
        },
        "anomaly_class_breakdown": breakdown
    }

    # Save to JSON
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print Formatted Report
    print("=" * 65)
    print("      DCTO ANOMALY DETECTOR BENCHMARK VALIDATION REPORT")
    print("=" * 65)
    print(f"Evaluated Cycles:      {num_cycles - warmup_cutoff} (post-warmup)")
    print(f"Evaluated Events:      {total_eval_events}")
    print(f"Ground Truth Anomalies:{len(gt_set)}")
    print(f"Detected Anomalies:    {len(detected_set)}")
    print("-" * 65)
    print(f"Precision:             {precision * 100:.2f}%  (Target: >= 88.0%)")
    print(f"Recall (Sensitivity):  {recall * 100:.2f}%  (Target: >= 90.0%)")
    print(f"F1-Score:              {f1:.4f}")
    print(f"False Positive Rate:   {fpr * 100:.2f}%  (Target: <= 2.5%)")
    print("-" * 65)
    print("Class Recall Breakdown:")
    for c_name, data in breakdown.items():
        print(f"  - {c_name:<18}: {data['recall']*100:.1f}% ({data['detected_count']}/{data['ground_truth_count']})")
    print("=" * 65)
    print(f"Metrics saved to: {metrics_file}\n")

    return results

if __name__ == "__main__":
    run_benchmark()
