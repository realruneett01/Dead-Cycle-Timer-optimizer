"""Generates high-resolution visualization charts for the repository documentation."""
import os
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Set clean professional styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#D1D5DB'
plt.rcParams['axes.linewidth'] = 0.8

ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def generate_telemetry_and_anomalies_chart():
    """Generates a 3-panel visualization showing telemetry time-series, waterfall, and hydraulic curves."""
    fig = plt.figure(figsize=(15, 10), facecolor='#FAFAFA', dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.25)
    
    # ---------------- PANEL A: Phase Duration Scatter & Baseline Drift ----------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#FFFFFF')
    
    np.random.seed(42)
    n_cycles = 120
    nominal = 2.80
    sigma = 0.10
    
    # Generate baseline normal data
    durations = [nominal + np.random.normal(0, sigma) for _ in range(n_cycles)]
    anom_indices = []
    anom_types = []
    
    # Inject valve overlap at cycles 22, 58, 95
    for c, delay in [(22, 0.45), (58, 0.52), (95, 0.48)]:
        durations[c] += delay
        anom_indices.append(c)
        anom_types.append("VALVE_OVERLAP")
        
    # Inject micro-stalls at cycles 35, 78, 110
    for c, delay in [(35, 0.85), (78, 1.10), (110, 0.92)]:
        durations[c] += delay
        anom_indices.append(c)
        anom_types.append("MICRO_STALL")
        
    # Inject creeping wear drift sequence from cycle 40 to 65
    for idx, c in enumerate(range(40, 65)):
        drift = 0.20 * nominal + (idx * 0.008 * nominal)
        durations[c] += drift
        anom_indices.append(c)
        anom_types.append("CREEPING_WEAR")
        
    cycles = np.arange(1, n_cycles + 1)
    durations = np.array(durations)
    
    # Plot baseline band
    ax1.axhline(nominal, color='#10B981', linestyle='--', linewidth=1.5, label='Nominal Baseline (2.80s)')
    threshold = nominal + 2.75 * sigma
    ax1.axhline(threshold, color='#F59E0B', linestyle=':', linewidth=1.5, label='Detection Threshold (+2.75σ = 3.08s)')
    ax1.fill_between([1, n_cycles], nominal - 2*sigma, nominal + 2*sigma, color='#10B981', alpha=0.08, label='Normal Operating Band (±2σ)')
    
    # Normal points
    normal_mask = np.ones(n_cycles, dtype=bool)
    normal_mask[anom_indices] = False
    ax1.scatter(cycles[normal_mask], durations[normal_mask], color='#3B82F6', alpha=0.75, s=28, edgecolors='none', label='Normal Cycle')
    
    # Anomaly points categorized
    for idx, atype in zip(anom_indices, anom_types):
        if atype == "VALVE_OVERLAP":
            ax1.scatter(cycles[idx], durations[idx], color='#D97706', s=55, marker='^', zorder=5)
        elif atype == "MICRO_STALL":
            ax1.scatter(cycles[idx], durations[idx], color='#EF4444', s=70, marker='X', zorder=5)
        elif atype == "CREEPING_WEAR":
            ax1.scatter(cycles[idx], durations[idx], color='#EA580C', s=35, marker='o', zorder=4)

    # Proxy markers for legend
    ax1.scatter([], [], color='#D97706', marker='^', s=55, label='Valve Overlap Delay (+0.35s–0.70s)')
    ax1.scatter([], [], color='#EF4444', marker='X', s=70, label='Micro-Stall / Stick-Slip (+0.50s–1.25s)')
    ax1.scatter([], [], color='#EA580C', marker='o', s=35, label='Creeping Seal Wear Drift (+20%–38%)')
    
    # Annotate Changepoint detection
    ax1.annotate('Pelt Changepoint Detected\n(Regime Break: Wear Sequence)', xy=(40, 3.42), xytext=(32, 4.40),
                 arrowprops=dict(facecolor='#1F2937', shrink=0.08, width=1.2, headwidth=6),
                 fontsize=8.5, fontweight='bold', color='#1F2937', ha='center',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='#FEF3C7', edgecolor='#F59E0B', alpha=0.95))

    ax1.set_title("A. Real-Time Telemetry & Multi-Tier Anomaly Detection Timeline (container_shift_close)", fontsize=12.5, fontweight='bold', pad=10, color='#111827')
    ax1.set_xlabel("Press Cycle Sequence Number", fontsize=10, labelpad=6)
    ax1.set_ylabel("Phase Duration (Seconds)", fontsize=10, labelpad=6)
    ax1.set_xlim(1, n_cycles)
    ax1.set_ylim(2.4, 4.85)
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=8.2, ncol=2)

    # ---------------- PANEL B: Cycle Phase Waterfall ----------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#FFFFFF')
    
    phases = [
        "1. Decompression", "2. Container Open", "3. Shear Stroke", "4. Die Slide",
        "5. Billet Load", "6. Container Close", "7. Rapid Advance"
    ]
    nom_durations = [2.2, 3.0, 2.5, 1.8, 3.2, 2.8, 2.5]
    excess_delays = [0.0, 0.42, 0.85, 0.0, 0.60, 0.35, 0.0]
    
    y_pos = np.arange(len(phases))[::-1]
    
    ax2.barh(y_pos, nom_durations, height=0.55, color='#2563EB', alpha=0.85, label='Nominal Duration')
    ax2.barh(y_pos, excess_delays, left=nom_durations, height=0.55, color='#EF4444', alpha=0.90, label='Recoverable Delay')
    
    for i, (nom, exc) in enumerate(zip(nom_durations, excess_delays)):
        idx = len(phases) - 1 - i
        total = nom + exc
        if exc > 0:
            ax2.text(total + 0.12, idx, f"{total:.2f}s (+{exc:.2f}s)", va='center', fontsize=8.5, fontweight='bold', color='#B91C1C')
        else:
            ax2.text(total + 0.12, idx, f"{total:.2f}s", va='center', fontsize=8.5, color='#4B5563')

    ax2.set_title("B. Dead-Cycle Phase Waterfall (Latest Cycle vs. Recoverable Delay)", fontsize=11, fontweight='bold', pad=10, color='#111827')
    ax2.set_xlabel("Duration (Seconds)", fontsize=9.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(phases, fontsize=9)
    ax2.set_xlim(0, 5.5)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.35)
    ax2.legend(loc='lower right', framealpha=0.95, fontsize=8.5)

    # ---------------- PANEL C: Hydraulic Pressure & Valve Command Curve ----------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#FFFFFF')
    
    t = np.linspace(0, 18, 400)
    # Synthetic realistic pressure curve across dead cycle
    p = np.piecewise(t, [
        (t >= 0) & (t < 2.2),
        (t >= 2.2) & (t < 5.2),
        (t >= 5.2) & (t < 7.7),
        (t >= 7.7) & (t < 9.5),
        (t >= 9.5) & (t < 12.7),
        (t >= 12.7) & (t < 15.5),
        (t >= 15.5) & (t <= 18.0)
    ], [
        lambda x: 280.0 * np.exp(-1.5 * x) + 15.0,     # Decompression drop
        lambda x: 20.0 + 35.0 * np.sin((x-2.2)/3.0 * np.pi), # Container open
        lambda x: 40.0 + 145.0 * ((x-5.2)/2.5)**2,    # Shear cut spike
        lambda x: 25.0 + 10.0 * np.sin((x-7.7)/1.8 * np.pi), # Die slide
        lambda x: 20.0 + 30.0 * ((x-9.5)/3.2),         # Billet load
        lambda x: 30.0 + 120.0 * ((x-12.7)/2.8)**1.5,  # Container close seal
        lambda x: 35.0 + 35.0 * ((x-15.5)/2.5)         # Rapid advance
    ])
    
    valve = np.piecewise(t, [
        (t >= 0) & (t < 2.2),
        (t >= 2.2) & (t < 5.2),
        (t >= 5.2) & (t < 7.7),
        (t >= 7.7) & (t < 9.5),
        (t >= 9.5) & (t < 12.7),
        (t >= 12.7) & (t < 15.5),
        (t >= 15.5) & (t <= 18.0)
    ], [
        lambda x: 45.0,
        lambda x: 80.0,
        lambda x: 92.0,
        lambda x: 60.0,
        lambda x: 75.0,
        lambda x: 85.0,
        lambda x: 95.0
    ])

    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(t, p, color='#0284C7', linewidth=2.0, label='Main Cylinder Pressure (bar)')
    line2 = ax3_twin.plot(t, valve, color='#8B5CF6', linestyle='-.', linewidth=1.5, label='Proportional Valve Spool (%)')
    
    ax3.set_title("C. Hydraulic Pressure (bar) & Valve Spool (%) Telemetry", fontsize=11, fontweight='bold', pad=10, color='#111827')
    ax3.set_xlabel("Dead-Cycle Time (Seconds)", fontsize=9.5)
    ax3.set_ylabel("Pressure (bar)", color='#0284C7', fontsize=9.5)
    ax3_twin.set_ylabel("Valve Position (%)", color='#8B5CF6', fontsize=9.5)
    ax3.set_ylim(0, 315)
    ax3_twin.set_ylim(0, 105)
    ax3.grid(True, linestyle='--', alpha=0.35)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right', framealpha=0.95, fontsize=8.5)

    plt.tight_layout()
    out_path = ASSETS_DIR / "telemetry_and_anomalies.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

def generate_benchmark_metrics_chart():
    """Generates a 3-panel visualization showing Confusion Matrix, Class Sensitivity, and Economic ROI."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor='#FAFAFA', dpi=300)
    
    # ---------------- PANEL 1: Confusion Matrix ----------------
    ax1 = axes[0]
    ax1.set_facecolor('#FFFFFF')
    
    # Evaluated benchmark numbers (600 cycles, 4600 events)
    cm = np.array([[293, 38], [4, 4265]])
    labels = [["TP: 293\n(6.4%)", "FP: 38\n(0.8%)"], ["FN: 4\n(0.1%)", "TN: 4,265\n(92.7%)"]]
    
    im = ax1.imshow(cm, cmap='Blues', interpolation='nearest', vmin=0, vmax=4500)
    
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > 2000 else "#111827"
            ax1.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=11, fontweight='bold')

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Predicted Delay", "Predicted Normal"], fontsize=9.5)
    ax1.set_yticklabels(["Actual Delay", "Actual Normal"], fontsize=9.5)
    ax1.set_title("A. Benchmark Confusion Matrix\n(Precision: 88.52% | Recall: 98.65%)", fontsize=11, fontweight='bold', pad=10, color='#111827')

    # ---------------- PANEL 2: Per-Class Recall Breakdown ----------------
    ax2 = axes[1]
    ax2.set_facecolor('#FFFFFF')
    
    classes = ["Creeping Seal\nWear Drift", "Micro-Stall /\nStick-Slip", "Valve Overlap\nInterlock Delay"]
    recalls = [100.0, 94.7, 89.7]
    colors = ['#10B981', '#3B82F6', '#F59E0B']
    
    bars = ax2.bar(classes, recalls, color=colors, width=0.55, edgecolor='#D1D5DB', linewidth=0.8)
    
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#111827')

    ax2.axhline(90.0, color='#EF4444', linestyle='--', linewidth=1.2, label='Target Recall (90.0%)')
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Detection Sensitivity / Recall (%)", fontsize=9.5)
    ax2.set_title("B. Class-Specific Anomaly Recall\n(Zero Missed Wear Cycles)", fontsize=11, fontweight='bold', pad=10, color='#111827')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax2.legend(loc='lower right', framealpha=0.95, fontsize=8.5)

    # ---------------- PANEL 3: Economic Capacity Recovery ----------------
    ax3 = axes[2]
    ax3.set_facecolor('#FFFFFF')
    
    categories = ["Direct Press\nCost Saved", "Tonnage\nMargin Gain", "Total Annual\nEconomic Value"]
    amounts = [122.55, 237.15, 359.70] # in k EUR
    colors_econ = ['#2563EB', '#0D9488', '#059669']
    
    bars_e = ax3.bar(categories, amounts, color=colors_econ, width=0.55, edgecolor='#D1D5DB', linewidth=0.8)
    
    for bar, val in zip(bars_e, amounts):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, h + 6.0, f"€{val:,.1f}k\n(+{val*1000/950:.0f} hrs)", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#111827')

    ax3.set_ylim(0, 420)
    ax3.set_ylabel("Annualized Financial Recovery (k€ / Year)", fontsize=9.5)
    ax3.set_title("C. Plant Capacity & Margin ROI\n(+129 Machine Hrs / +527 MT Output)", fontsize=11, fontweight='bold', pad=10, color='#111827')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.35)

    plt.tight_layout()
    out_path = ASSETS_DIR / "benchmark_metrics.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

def generate_architecture_diagram():
    """Generates a clean, modern vector-style pipeline architecture infographic."""
    fig, ax = plt.subplots(figsize=(15, 5.2), facecolor='#FAFAFA', dpi=300)
    ax.set_facecolor('#FAFAFA')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.axis('off')

    stages = [
        ("1. Press Simulation\n& Fault Injection", 
         "• 8 Discrete Kinematic Phases\n• Stochastic Gaussian Jitter\n• Valve Overlap & Micro-Stall\n• Creeping Wear Degradation\n• Ground-Truth Logger (CSV)",
         "#EFF6FF", "#2563EB", 1.8),
        ("2. Industrial OT Bridge\n(IEC 62541 OPC-UA)", 
         "• Asynchronous asyncua Server\n• ISA-95 Node Namespace\n• CurrentPhase, Duration Tags\n• Pressure & Valve Telemetry\n• Real-Time Edge Subscriptions",
         "#F0FDF4", "#059669", 5.5),
        ("3. Multi-Tier Anomaly\nDetection Engine", 
         "• Tier 1: Robust Z-Score (MAD)\n• Tier 2: Ruptures Pelt (RBF)\n• Tier 3: Gaussian HMM (3-State)\n• 98.65% Recall | 88.52% Precision\n• Anti-Poisoning Quarantine",
         "#FEF3C7", "#D97706", 9.2),
        ("4. Executive & Engineering\nDashboard (Streamlit)", 
         "• Live Phase Waterfall Gantt\n• Sub-Second Stall Diagnostics\n• Root-Cause Attribution\n• Plant Financial & OEE Modeler\n• Assumption-Based Transparent UI",
         "#F5F3FF", "#7C3AED", 12.9)
    ]

    for title, desc, bg_color, border_color, x_center in stages:
        # Draw box
        box = patches.FancyBboxPatch((x_center - 1.6, 0.6), 3.2, 3.8,
                                     boxstyle="round,pad=0.15",
                                     facecolor=bg_color,
                                     edgecolor=border_color,
                                     linewidth=2.0)
        ax.add_patch(box)
        
        # Title banner
        ax.text(x_center, 3.9, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=border_color)
        
        # Divider line
        ax.plot([x_center - 1.35, x_center + 1.35], [3.35, 3.35], color=border_color, alpha=0.3, linewidth=1.2)
        
        # Description
        ax.text(x_center - 1.35, 2.1, desc, ha='left', va='center', fontsize=8.8, color='#374151', linespacing=1.6)

    # Arrows connecting stages
    arrow_positions = [(3.45, 2.5), (7.15, 2.5), (10.85, 2.5)]
    for x_arr, y_arr in arrow_positions:
        ax.annotate('', xy=(x_arr + 0.40, y_arr), xytext=(x_arr, y_arr),
                    arrowprops=dict(facecolor='#4B5563', edgecolor='none', width=2.5, headwidth=8, headlength=8))

    plt.tight_layout()
    out_path = ASSETS_DIR / "architecture_diagram.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    generate_telemetry_and_anomalies_chart()
    generate_benchmark_metrics_chart()
    generate_architecture_diagram()
