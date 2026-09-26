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

from collections import deque
from detector.detector_service import DetectorService
from detector.cusum_detector import PageCusumDetector
from simulator.anomaly_injector import AnomalyInjector
from simulator.press_config import PressConfig
from simulator.press_state_machine import PressStateMachine

def run_benchmark(
    num_cycles: int = 600,
    anomaly_prob: float = 0.12,
    seed: int = 42,
    threshold_z: float = 2.75,
    cusum_alpha: float = 0.01,
    cusum_beta: float = 0.05
) -> dict:
    """
    Executes a benchmark simulation run, streams events through both DetectorService
    (Rolling MAD Baseline) and PageCusumDetector (Page's CUSUM grounded in SPRT),
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
    cusum_detector = PageCusumDetector(alpha=cusum_alpha, beta=cusum_beta, default_delta_sec=0.35, min_baseline_samples=15)

    # Rolling buffer for CUSUM baseline tracking
    cusum_histories = {name: deque(maxlen=40) for name in config.canonical_phase_names}
    for name, p in config.phases.items():
        cusum_histories[name].extend([p.nominal_sec] * 20)

    cusum_alarms = []

    print(f"\n--- Running Benchmark Simulation ({num_cycles} cycles, p={anomaly_prob}, threshold_z={threshold_z}) ---")
    
    # Process all cycles
    for cycle_id in range(1, num_cycles + 1):
        events = sm.run_cycle(cycle_id=cycle_id)
        for ev in events:
            # 1. Evaluate DetectorService (Rolling Baseline + Changepoint + HMM)
            detector.process_phase_event(
                cycle_id=ev.cycle_id,
                phase_name=ev.phase_name,
                duration=ev.duration_sec,
                is_dead_cycle=ev.is_dead_cycle
            )

            # 2. Evaluate Page's CUSUM Detector (Sequential SPRT Formulation)
            hist = cusum_histories[ev.phase_name]
            arr = np.array(list(hist))
            mu = float(np.median(arr))
            mad = float(np.median(np.abs(arr - mu)))
            sigma = max(1.4826 * mad, config.phases[ev.phase_name].std_dev_sec * 0.8)

            c_res = cusum_detector.evaluate_observation(
                cycle_id=ev.cycle_id,
                phase_name=ev.phase_name,
                duration=ev.duration_sec,
                baseline_mu=mu,
                baseline_sigma=sigma,
                sample_count=len(hist)
            )
            if c_res.is_alarm:
                cusum_alarms.append((ev.cycle_id, ev.phase_name))
            else:
                hist.append(ev.duration_sec)

    # Load produced datasets
    df_gt = pd.read_csv(gt_file)
    df_telem = pd.read_csv(telem_file)

    # Exclude initial warmup period (first 25 cycles) for fair evaluation
    warmup_cutoff = 25
    df_gt_eval = df_gt[df_gt["cycle_id"] > warmup_cutoff].copy()
    df_telem_eval = df_telem[df_telem["cycle_id"] > warmup_cutoff].copy()

    # Ground truth mapping
    gt_set = set(zip(df_gt_eval["cycle_id"], df_gt_eval["phase_name"]))
    gt_type_map = dict(zip(zip(df_gt_eval["cycle_id"], df_gt_eval["phase_name"]), df_gt_eval["anomaly_type"]))
    all_events_set = set(zip(df_telem_eval["cycle_id"], df_telem_eval["phase_name"]))
    total_eval_events = len(all_events_set)

    # --- 1. Evaluate DetectorService ---
    detected_mask = df_telem_eval["is_anomaly"] == True
    t1_detected_set = set(zip(df_telem_eval.loc[detected_mask, "cycle_id"], df_telem_eval.loc[detected_mask, "phase_name"]))

    tp1 = len(t1_detected_set.intersection(gt_set))
    fp1 = len(t1_detected_set - gt_set)
    fn1 = len(gt_set - t1_detected_set)
    tn1 = total_eval_events - (tp1 + fp1 + fn1)

    prec1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0.0
    rec1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0.0
    f1_1 = 2 * (prec1 * rec1) / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0
    fpr1 = fp1 / (fp1 + tn1) if (fp1 + tn1) > 0 else 0.0

    # --- 2. Evaluate Page's CUSUM ---
    cusum_detected_set = {a for a in cusum_alarms if a[0] > warmup_cutoff}

    tp2 = len(cusum_detected_set.intersection(gt_set))
    fp2 = len(cusum_detected_set - gt_set)
    fn2 = len(gt_set - cusum_detected_set)
    tn2 = total_eval_events - (tp2 + fp2 + fn2)

    prec2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0.0
    rec2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0.0
    f1_2 = 2 * (prec2 * rec2) / (prec2 + rec2) if (prec2 + rec2) > 0 else 0.0
    fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0.0

    # Per-anomaly-type breakdown
    breakdown1 = {}
    breakdown2 = {}
    for anom_type in ["VALVE_OVERLAP", "MICRO_STALL", "CREEPING_WEAR"]:
        type_gt = {k for k, v in gt_type_map.items() if v == anom_type}
        type_total = len(type_gt)
        
        type_tp1 = len(type_gt.intersection(t1_detected_set))
        type_rec1 = type_tp1 / type_total if type_total > 0 else 0.0
        breakdown1[anom_type] = {"total": type_total, "detected": type_tp1, "recall": round(type_rec1, 4)}

        type_tp2 = len(type_gt.intersection(cusum_detected_set))
        type_rec2 = type_tp2 / type_total if type_total > 0 else 0.0
        breakdown2[anom_type] = {"total": type_total, "detected": type_tp2, "recall": round(type_rec2, 4)}

    results = {
        "benchmark_parameters": {
            "num_cycles": num_cycles,
            "warmup_cycles_excluded": warmup_cutoff,
            "anomaly_probability": anomaly_prob,
            "random_seed": seed,
            "threshold_z": threshold_z,
            "cusum_alpha": cusum_alpha,
            "cusum_beta": cusum_beta
        },
        "baseline_detector": {
            "confusion_matrix": {"TP": tp1, "FP": fp1, "FN": fn1, "TN": tn1},
            "precision": round(prec1, 4),
            "recall": round(rec1, 4),
            "f1_score": round(f1_1, 4),
            "false_positive_rate": round(fpr1, 4),
            "class_recall": breakdown1
        },
        "page_cusum_detector": {
            "confusion_matrix": {"TP": tp2, "FP": fp2, "FN": fn2, "TN": tn2},
            "precision": round(prec2, 4),
            "recall": round(rec2, 4),
            "f1_score": round(f1_2, 4),
            "false_positive_rate": round(fpr2, 4),
            "class_recall": breakdown2
        }
    }

    # Save to JSON
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print Formatted Comparison Report
    print("=" * 72)
    print("       DCTO ANOMALY DETECTOR BENCHMARK: BASELINE vs. PAGE'S CUSUM")
    print("=" * 72)
    print(f"Evaluated Cycles:      {num_cycles - warmup_cutoff} (warmup: {warmup_cutoff} excluded)")
    print(f"Evaluated Events:      {total_eval_events}")
    print(f"Ground Truth Anomalies:{len(gt_set)}")
    print("-" * 72)
    print(f"{'Metric':<22} | {'Rolling MAD Baseline':<20} | {'Page CUSUM (SPRT)':<20}")
    print("-" * 72)
    print(f"{'True Positives (TP)':<22} | {tp1:<20} | {tp2:<20}")
    print(f"{'False Positives (FP)':<22} | {fp1:<20} | {fp2:<20}")
    print(f"{'False Negatives (FN)':<22} | {fn1:<20} | {fn2:<20}")
    print(f"{'True Negatives (TN)':<22} | {tn1:<20} | {tn2:<20}")
    print("-" * 72)
    print(f"{'Precision':<22} | {prec1 * 100:.2f}%{'':<14} | {prec2 * 100:.2f}%{'':<14}")
    print(f"{'Recall (Sensitivity)':<22} | {rec1 * 100:.2f}%{'':<14} | {rec2 * 100:.2f}%{'':<14}")
    print(f"{'F1-Score':<22} | {f1_1:.4f}{'':<14} | {f1_2:.4f}{'':<14}")
    print(f"{'False Positive Rate':<22} | {fpr1 * 100:.2f}%{'':<14} | {fpr2 * 100:.2f}%{'':<14}")
    print("-" * 72)
    print("Per-Class Recall Breakdown:")
    for c_name in ["VALVE_OVERLAP", "MICRO_STALL", "CREEPING_WEAR"]:
        r1_str = f"{breakdown1[c_name]['recall']*100:.1f}% ({breakdown1[c_name]['detected']}/{breakdown1[c_name]['total']})"
        r2_str = f"{breakdown2[c_name]['recall']*100:.1f}% ({breakdown2[c_name]['detected']}/{breakdown2[c_name]['total']})"
        print(f"  - {c_name:<18} | {r1_str:<20} | {r2_str:<20}")
    print("=" * 72)
    print(f"Metrics saved to: {metrics_file}\n")

    return results

if __name__ == "__main__":
    run_benchmark()
