# ⚡ Dead-Cycle Time Optimizer (DCTO)
### Industrial Edge AI/ML: Real-Time Cycle Phase Telemetry, Sub-Second Micro-Stall Diagnostics, and Dead-Cycle Time Recovery for 15–30 MN Hydraulic Extrusion Presses

<div align="center">

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OPC-UA](https://img.shields.io/badge/IEC_62541-OPC--UA-005C8A?style=for-the-badge&logo=industrial-shields&logoColor=white)](https://opcfoundation.org/)
[![Tests](https://img.shields.io/badge/Pytest-12_Passed-10B981?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Precision](https://img.shields.io/badge/Precision-95.42%25_(Page_CUSUM)-2563EB?style=for-the-badge)](#-empirical-benchmark-validation-baseline-vs-pages-cusum)
[![Recall](https://img.shields.io/badge/Recall-98.32%25-059669?style=for-the-badge)](#-empirical-benchmark-validation-baseline-vs-pages-cusum)
[![F1-Score](https://img.shields.io/badge/F1_Score-0.9685-7C3AED?style=for-the-badge)](#-empirical-benchmark-validation-baseline-vs-pages-cusum)
[![FPR](https://img.shields.io/badge/FPR-0.33%25-10B981?style=for-the-badge)](#-empirical-benchmark-validation-baseline-vs-pages-cusum)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](dashboard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-6B7280?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Executive Summary & Industrial Context

In heavy industrial aluminum extrusion (15 MN – 50+ MN presses), plant throughput and operating margins are dictated by the cadence of the extrusion press. Every aluminum billet cycle splits into two fundamentally distinct phases:
1. **Extrusion Stroke (Productive):** The main ram drives a heated billet (~450–500°C) through tool steel dies under 250–315 bar hydraulic pressure (45 to 120+ seconds).
2. **Dead-Cycle Time (DCT / Auxiliary Stroke):** The non-productive mechanical repositioning sequence (14 to 22 seconds): Decompression, Container Shift Open, Butt Shear, Die Slide Indexing, Billet Loading, Container Shift Close, and Main Ram Rapid Advance.

```
+---------------------------------------------------------------------------------------------------------+
|                                        FULL BILLET PRESS CYCLE                                          |
+-----------------------------------------+---------------------------------------------------------------+
|       EXTRUSION STROKE (Productive)     |                   DEAD-CYCLE TIME (DCT) (Non-Productive)      |
|           45 - 120 seconds              |                        14 - 22 seconds                        |
|   (Billet pushed through die under P)   |  Decompress -> Container Open -> Shear -> Billet Load -> Close|
+-----------------------------------------+---------------------------------------------------------------+
                                                                 ▲
                                                  Target for AI/ML Optimization:
                                        Eliminate 1.0 - 2.5s of Hidden Micro-Stalls
```

### The Industrial Problem: Invisible Micro-Stalls & The SCADA Blind Spot
Traditional PLC alarm thresholds and SCADA supervisory systems only flag gross mechanical failures (e.g., `Container Retract Timeout > 6.0s`). They are **blind** to transient 200–800 ms hydraulic valve hesitations, stick-slip friction, and contact bounce:
- **Valve Spool Hesitation & Oil Varnish:** High-speed proportional directional valves (ISO VG 46 fluid) suffer sluggish spool movement during cold starts or thermal transients.
- **Micro-Stalls & Mechanical Stiction:** Debris or seal wear on container tie rods causes momentary 500–1200 ms halts during continuous motion strokes.
- **Creeping Mechanical Drift:** Cylinder seal blow-by and proportional relief valve pilot degradation introduce gradual duration inflation (+20%–40%) over weeks before an overt fault occurs.
- **Compounding Loss:** Losing an average of **1.5 seconds per dead-cycle** on a press cycling 43 times an hour across 7,200 annual operating hours bleeds **129 hours of machine availability annually (€122,550 direct cost / +527 metric tons of lost aluminum output)**.

---

## 🏗️ System Architecture

DCTO bridges operational technology (OT) and statistical sequential analysis through a modular, edge-native architecture designed for sub-millisecond execution and zero cloud dependency:

![System Architecture](assets/architecture_diagram.png)

```mermaid
flowchart LR
    subgraph S1["1. Simulation & Fault Engine"]
        SM[Press State Machine<br/>8 Kinematic Phases]
        INJ[Fault Injector<br/>Stalls, Overlaps, Wear]
        GT[(Ground Truth CSV<br/>Immutable Audit)]
        SM --> INJ --> GT
    end

    subgraph S2["2. Industrial OT Bridge"]
        SRV[OPC-UA Server<br/>IEC 62541 asyncua]
        TAG[ISA-95 Hierarchy<br/>Pressure, Valve, Phase]
        SRV --- TAG
    end

    subgraph S3["3. Statistical & Sequential Engine"]
        CUS[Primary: Page's CUSUM<br/>SPRT Bounded Decisions]
        ZSC[Tier 1: Rolling MAD Baseline<br/>Z-Score Outlier Quarantine]
        CPD[Tier 2: Ruptures Pelt<br/>RBF Kernel Changepoint]
        HMM[Tier 3: Gaussian HMM<br/>Latent State Viterbi]
        CUS & ZSC & CPD & HMM --> FUS[Decision Fusion]
    end

    subgraph S4["4. Dashboard & Diagnostics"]
        UI[Streamlit App]
        WF[Phase Waterfall Gantt]
        DG[Micro-Stall Diagnostics]
        ROI[Plant OEE Financial Modeler]
        UI --> WF & DG & ROI
    end

    S1 -->|Live Telemetry Events| S2
    S2 -->|Subscribed Streams| S3
    S3 -->|Validated Diagnostics| S4
```

---

## 📈 Real-Time Telemetry & Detection Dynamics

The figure below illustrates the core detection capabilities running against press telemetry:

![Real-Time Telemetry and Multi-Tier Detection](assets/telemetry_and_anomalies.png)

### Key Insights from the Telemetry Profile:
- **Panel A (Phase Duration Scatter & Multi-Tier Alerts):** Demonstrates real-time detection of isolated valve overlap delays (triangles), severe micro-stalls (crosses), and the onset of a 25-cycle creeping seal wear degradation sequence (circles). The **Pelt changepoint detector** pinpoints the exact cycle (Cycle 40) where hydraulic friction regime broke from nominal baseline.
- **Panel B (Dead-Cycle Phase Waterfall):** Decomposes each dead-cycle step into its nominal duration (blue) and recoverable excess delay (red). Operators immediately identify that the Shear Stroke (+0.85s) and Billet Loader (+0.60s) contributed the bulk of lost cycle time.
- **Panel C (Hydraulic Cylinder Pressure & Valve Spool):** Continuous 10 Hz synchronous telemetry depicting the decompression pressure drop (280 bar $\rightarrow$ 15 bar), shear stroke cutting pressure spike (185 bar), container seal lockup (150 bar), and proportional valve command modulation (45% $\rightarrow$ 95%).

---

## 📊 Datasets & Physical Calibration

DCTO uses a two-component data foundation combining **physics-informed stochastic press simulation** and an **immutable ground-truth fault injection audit trail**.

### 1. Simulated Press Kinematic Dataset
To reflect physical dynamics of 15 MN – 30 MN direct-drive oil hydraulic presses, telemetry is generated from physical kinematics calibrated against standard press timing:

| Phase Index | Phase Name | Nominal Duration ($t_{nom}$) | Gaussian Jitter ($\sigma$) | Actuator / Subsystem | Hydraulic Operating Condition |
|---|---|---|---|---|---|
| **0** | `decompression` | **2.20 s** | $\pm 0.08\text{ s}$ | Main cylinder decompression valves | Pressure drop: $280 \rightarrow 15\text{ bar}$ |
| **1** | `container_shift_open` | **3.00 s** | $\pm 0.12\text{ s}$ | Twin container shift cylinders (400 mm) | Flow: 80% proportional valve |
| **2** | `shear_stroke` | **2.50 s** | $\pm 0.10\text{ s}$ | Vertical hydraulic butt shear | Cutting spike: 185 bar |
| **3** | `die_slide_check` | **1.80 s** | $\pm 0.06\text{ s}$ | Lateral die cassette indexing slide | Proportional position control |
| **4** | `billet_load` | **3.20 s** | $\pm 0.14\text{ s}$ | Overhead pivoting billet loader arm | Mechanical swing & grip |
| **5** | `container_shift_close` | **2.80 s** | $\pm 0.10\text{ s}$ | Twin container shift cylinders | Clamping against die bolster |
| **6** | `rapid_advance` | **2.50 s** | $\pm 0.10\text{ s}$ | Main ram pre-fill & side cylinders | Pre-fill stroke forward to billet |
| **7** | `extrusion_stroke` | **55.00 s** | $\pm 3.50\text{ s}$ | Main cylinder + side cylinders | Full extrusion work (250–315 bar) |

- **Sampling Rates:** Discrete event timestamps recorded at sub-millisecond precision; continuous pressure and valve spool positions sampled at 10 Hz (100 ms intervals).
- **Benchmark Evaluation Volume:** 600 full press cycles, producing **4,800 discrete phase events** and **86,400 synchronous hydraulic telemetry observations**.

### 2. Injected Ground-Truth Fault Dataset (`data/ground_truth.csv`)
Anomalies are injected using controlled stochastic and deterministic fault schedules to establish an objective evaluation benchmark:

| Anomaly Mode | Injected Magnitude | Typical Physical Root Cause | Target Detector Tier |
|---|---|---|---|
| **`VALVE_OVERLAP`** | **+0.35 s to +0.70 s** | Directional valve spool stick-slip, PLC digital output interlock delay, sensor contact bounce | **Page's CUSUM (SPRT)** |
| **`MICRO_STALL`** | **+0.50 s to +1.25 s** | Hydraulic fluid contamination, guide rail stiction, mechanical hesitation during travel | **Page's CUSUM & Tier 1 MAD** |
| **`CREEPING_WEAR`** | **+20% to +38% drift** across 25–40 cycles | Progressive cylinder piston seal blow-by, internal valve leakage, proportional relief pilot wear | **Tier 2 (Ruptures Pelt Changepoint)** |

Every injected anomaly is logged with cycle index, phase name, true excess delay, and fault category into `data/ground_truth.csv`, ensuring zero data leakage and reproducible evaluation.

---

## 🎯 Empirical Benchmark Validation: Baseline vs. Page's CUSUM

Both detectors were evaluated head-to-head on the 600-cycle benchmark dataset (575 post-warmup cycles, 4,600 evaluated events, 297 ground-truth anomalies). Run the validation script live at any time via:
```powershell
python tests/validate_detector.py
```

![Benchmark Metrics and ROI](assets/benchmark_metrics.png)

### Side-by-Side Performance Scorecard

| Evaluation Metric | Rolling MAD Baseline | **Page's CUSUM (SPRT-Grounded)** | Delta / Improvement | Status |
|---|---|---|---|---|
| **True Positives (TP)** | 293 | **292** | -1 | ✅ High Sensitivity |
| **False Positives (FP)** | 38 | **14** | **-63.2% Reduction** | 🛡️ Slashes False Alarms |
| **False Negatives (FN)** | 4 | **5** | +1 | ✅ Minimal Missed Faults |
| **True Negatives (TN)** | 4,265 | **4,289** | +24 | ✅ High Specificity |
| **Precision** | 88.52% | **95.42%** | **+6.90%** | ✅ **SUPERIOR** |
| **Recall (Sensitivity)** | **98.65%** | **98.32%** | -0.33% | ✅ **PASSED** |
| **F1-Score** | 0.9331 | **0.9685** | **+0.0354** | ✅ **PASSED** |
| **False Positive Rate (FPR)** | 0.88% | **0.33%** | **-62.5% Reduction** | ✅ **PASSED** |

### Per-Class Detection Recall Breakdown
- 🟢 **Creeping Wear Sequences (`CREEPING_WEAR`):** **100.0%** (249 / 249 cycles captured by both)
- 🔵 **Valve Overlap Delays (`VALVE_OVERLAP`):** **89.7%** (26 / 29 delays detected by both)
- 🟡 **Micro-Stalls & Stick-Slip (`MICRO_STALL`):** **89.5% CUSUM** (17 / 19) vs. **94.7% Baseline** (18 / 19)

> [!NOTE]
> **Why 3 Valve Overlap delays were missed:** In phases with higher inherent baseline variance (`billet_load`, $\sigma = 0.14\text{s}$), a small injected delay of $+0.35\text{s}$ coincided with a negative stochastic draw ($-0.11\text{s}$), leaving an effective excess delay of only $+0.24\text{s}$. On pure phase-duration residuals, that cannot be statistically separated from normal machine jitter without lowering the decision boundary and inflating false alarms.

### Plant-Wide Capacity & Economic Recovery
Assuming standard commercial operating parameters for a single 28 MN press:
- **Direct Press Operating Cost:** €950 / hour
- **Extrusion Tonnage:** 95 kg billet weight, 43 cycles/hour nominal cadence
- **Annual Dead-Cycle Reduction:** 1.5 seconds average recovered delay per cycle
- **Machine Availability Reclaimed:** **129.0 press operating hours / year**
- **Direct Machine Cost Savings:** **€122,550 / year**
- **Extrusion Capacity Gain:** **+5,547 billets (+527 Metric Tons)** of profile production
- **Total Annualized Economic Value:** **€359,700 / year / press**

---

## 🧮 Mathematical & Algorithmic Methodology

### 1. Primary Detector: Page's CUSUM Sequential Test (Page, 1954; Wald, 1945)
Rather than choosing an ad-hoc threshold by eye, **Page's Cumulative Sum (CUSUM) test** derives its decision boundary directly from pre-stated, mathematically defensible error bounds.

Given:
- Null Hypothesis $H_0: x_n \sim \mathcal{N}(\mu_0, \sigma_0^2)$ (nominal operating cycle)
- Alternative Hypothesis $H_1: x_n \sim \mathcal{N}(\mu_0 + \delta, \sigma_0^2)$ (delayed cycle with physical shift $\delta \ge 0.35\text{s}$)
- Target Type I error bound (false-alarm probability): $\alpha = 0.01$ (1%)
- Target Type II error bound (missed-detection probability): $\beta = 0.05$ (5%)

From Wald's SPRT log-likelihood ratio, the optimal sequential increment is:

$$s_n = \frac{\delta}{\sigma_0^2} \left( (x_n - \mu_0) - \frac{\delta}{2} \right)$$

Page's CUSUM accumulates positive evidence and re-arms after reaching zero:

$$S_0 = 0, \quad S_n = \max(0, \, S_{n-1} + s_n)$$

The decision threshold $h$ is derived from the SPRT stopping boundary:

$$h = \ln\left(\frac{1 - \beta}{\alpha}\right) = \ln\left(\frac{0.95}{0.01}\right) = \ln(95) \approx 4.554$$

* **Decision Rule:** When $S_n \ge h$, an alarm is asserted, and $S_n$ is reset to zero to continue monitoring subsequent cycles.

### 2. Tier 1: Outlier-Resistant Rolling Baseline (Median Absolute Deviation)
Tracks running baseline $\mu_0$ and robust scale $\sigma_0$ using the Hampel Median Absolute Deviation (MAD) over a rolling window $W = 40$:

$$\tilde{x} = \text{median}(X_W), \quad \text{MAD} = \text{median}\left(\left| x_i - \tilde{x} \right|\right), \quad \hat{\sigma}_{\text{robust}} = 1.4826 \times \text{MAD}$$

$$Z_{\text{robust}}(x) = \frac{x - \tilde{x}}{\hat{\sigma}_{\text{robust}}}$$

### 3. Tier 2: Penalized Cost Changepoint Detection (Pelt)
Distinguishes transient micro-stalls from progressive mechanical degradation using non-parametric changepoint search via the `ruptures` library:

$$\min_{\mathcal{T}} \sum_{k=0}^{K} \mathcal{C}\left(y_{\tau_k : \tau_{k+1}}\right) + \beta K$$

- **Algorithm:** Pruned Exact Linear Time (**Pelt**) in $\mathcal{O}(N)$ computation time with an **RBF** kernel.
- **Persistent Wear Latch:** When a statistically significant changepoint is confirmed across consecutive cycles, the phase is tagged with `CREEPING_WEAR` until recalibration.

### 4. Tier 3: Gaussian Hidden Markov Model (HMM)
A 3-state Gaussian HMM decodes unobserved machine health states via the Viterbi dynamic programming algorithm (State 0: Healthy, State 1: Creeping Wear, State 2: Transient Stall).

---

## 📁 Repository Structure

```
Dead-Cycle-Timer-optimizer/
├── assets/                                     # High-resolution architectural & benchmark figures
│   ├── architecture_diagram.png                # 4-stage pipeline infographic
│   ├── telemetry_and_anomalies.png             # Scatter, waterfall & hydraulic telemetry
│   └── benchmark_metrics.png                   # Confusion matrix, class recall & economic ROI
│
├── simulator/                                  # Press cycle & anomaly generation
│   ├── press_config.py                         # 28 MN press nominal timing & physical parameters
│   ├── press_state_machine.py                  # 8-phase state machine with sensor signals
│   └── anomaly_injector.py                     # Deterministic fault injector & ground-truth logger
│
├── opcua/                                      # Industrial OPC-UA telemetry server
│   ├── opcua_nodes.py                          # ISA-95 hierarchical node definitions
│   ├── opcua_server.py                         # Async OPC-UA server (asyncua)
│   └── test_client.py                          # Verification subscription client
│
├── detector/                                   # Anomaly detection engines
│   ├── cusum_detector.py                       # Page's CUSUM sequential test (SPRT-grounded)
│   ├── rolling_baseline.py                     # Tier 1: Z-score & MAD rolling statistics
│   ├── changepoint_detector.py                 # Tier 2: ruptures Pelt changepoint analysis
│   ├── hmm_detector.py                         # Tier 3: hmmlearn Gaussian HMM classifier
│   └── detector_service.py                     # Unified multi-tier detection runner
│
├── dashboard/                                  # Streamlit & Plotly interactive UI
│   ├── dashboard_app.py                        # Multi-tab Streamlit dashboard application
│   ├── components.py                           # Reusable Plotly charts (Gantt, scatter, waterfall)
│   └── oee_calculator.py                       # OEE, tonnage, and financial ROI calculator
│
├── tests/                                      # Automated Pytest test suite
│   ├── smoke_test.py                           # Environment & import verification
│   ├── test_simulator.py                       # Timing, bounds, and ground truth tests
│   ├── test_opcua.py                           # Server & client subscription integration tests
│   ├── test_detector.py                        # Statistical, CUSUM & algorithm unit tests
│   ├── test_end_to_end.py                      # Full pipeline integration test
│   └── validate_detector.py                    # Side-by-side benchmark comparison script
│
├── requirements.txt                            # Pinned Python dependencies
├── .gitignore                                  # Git ignore specifications
├── LICENSE                                     # MIT License
└── README.md                                   # Technical documentation & project portfolio
```

---

## 🚀 Quickstart & Execution Guide

### 1. Environment Setup
Clone the repository and initialize the Python virtual environment:
```powershell
# Clone and enter repository
git clone https://github.com/realruneett01/Dead-Cycle-Timer-optimizer.git
cd Dead-Cycle-Timer-optimizer

# Create and activate Python 3.11 virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
Execute the 12 unit and integration tests covering simulation, CUSUM detector, OPC-UA protocol exchange, and statistical algorithms:
```powershell
pytest -v
```
*Expected: 12 passed in ~6 seconds.*

### 3. Run Benchmark Validation Harness
Execute the 600-cycle validation harness to reproduce the side-by-side comparison report:
```powershell
python tests/validate_detector.py
```
*Prints comparative confusion matrices, Precision (95.42% CUSUM vs. 88.52% Baseline), Recall, and per-class breakdowns.*

### 4. Run OPC-UA Industrial Server & Edge Client
Start the asynchronous IEC 62541 OPC-UA server publishing ISA-95 node telemetry:
```powershell
# Terminal 1: Start OPC-UA server at 20x real-time simulation speed (module syntax)
python -m opcua.opcua_server --speedup 20.0

# Terminal 2: Connect real-time subscription test client
python -m opcua.test_client --events 30
```
*(Direct script syntax `python opcua/opcua_server.py` is also supported).*

### 5. Launch the Executive & Diagnostics Dashboard
Run the Streamlit interactive dashboard:
```powershell
streamlit run dashboard/dashboard_app.py
```
Open **`http://localhost:8501`** to interact with:
- **Tab 1: Live Phase Waterfall Gantt** — Monitor real-time cycle phase durations against target baselines.
- **Tab 2: Anomaly Stream & Diagnostics** — Filter by fault class (`VALVE_OVERLAP`, `MICRO_STALL`, `CREEPING_WEAR`) and inspect phase-by-phase box plots and drift curves.
- **Tab 3: Plant OEE & ROI Modeler** — Adjust press operating costs, billet sizes, and shift schedules to model customized financial and tonnage capacity gains.

---

## 🏭 Edge Deployment & Measured Latency Benchmarks

Benchmarked on standard x86 CPU without specialized hardware acceleration:
- **Page's CUSUM Evaluation Latency:** Measured execution time averaged **1.78 µs** per event (p99 = 3.70 µs).
- **Full Detection Pipeline Latency:** Measured execution time averaged **0.36 ms** per event (p99 = 0.72 ms), well within 1–10 ms PLC cycle boundaries.
- **Memory Footprint:** Observed continuous background execution memory stabilizes at **~65 MB RSS**.
- **Integration Architecture:** Designed for supervisory edge deployment alongside plant PLCs via standard IEC 62541 OPC-UA client/server subscriptions.

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
