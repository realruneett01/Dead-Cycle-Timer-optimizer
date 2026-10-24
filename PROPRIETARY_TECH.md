# What is actually proprietary in DCTO

An audit of the codebase, separating what is **ours** from what is **published science or a
third-party library**. Use this when deciding what to protect (private repo, trade secret,
patent counsel) and what to describe publicly.

## Not proprietary (public, do not claim)

| Component | Source |
|---|---|
| Page's CUSUM, Wald's SPRT threshold `h = ln((1-β)/α)` | Page 1954, Wald 1945 |
| Pelt changepoint search | `ruptures` library (Killick et al. 2012) |
| Gaussian HMM, Viterbi decoding | `hmmlearn` |
| Median/MAD robust z-score | textbook |
| OPC-UA transport | `asyncua`, IEC 62541 |

Anyone can reimplement each of these from the literature.

## Proprietary: the integration, calibration, and domain knowledge

| # | Asset | Where | Why it is hard to copy |
|---|---|---|---|
| 1 | **Fusion rule across heterogeneous detectors**: `CUSUM ∨ drift-latch ∨ (HMM-state ≥ 1 ∧ Tier-1 excess > 0.20 s)`, tuned so fused recall equals the best single detector while precision stays near CUSUM's | `detector/detector_service.py` `_fuse_anomaly_decision` | Thresholds come from the 600-cycle benchmark; the public papers do not give them |
| 2 | **Self-masking-proof CUSUM baseline**: alarmed cycles are excluded from the CUSUM reference window, with a σ floor of 0.8 × the physical per-phase jitter | `_evaluate_cusum`, `_cusum_baseline` | A naive rolling baseline absorbs a creeping fault and stops alarming; this is the fix |
| 3 | **Drift latch with gated Pelt**: Pelt only runs after ≥ 3 consecutive Tier-1 anomalies and ≥ 20 samples, latches `CREEPING_WEAR`, and releases when duration returns to within 0.15 s of baseline | `_update_drift_latch` | Cuts Pelt cost and false latches; needs the empirical wear signatures below |
| 4 | **Per-phase HMM initialised from physical priors** at distinct means (healthy / wear / stall) so EM cannot collapse two fault states into one | `detector/hmm_detector.py` `initialize_with_priors` | Standard HMM training collapses without this on near-unimodal press data |
| 5 | **Physics-calibrated 28 MN press model**: 8-phase timings, jitter σ, pressure/valve-spool profiles per phase | `simulator/press_config.py`, `press_state_machine.py` | Domain data, not derivable from papers |
| 6 | **Fault taxonomy and magnitudes** (valve overlap +0.35–0.70 s, micro-stall +0.50–1.25 s, creeping wear +20–38 % over 25–40 cycles) with leakage-free ground truth | `simulator/anomaly_injector.py` | Defines the benchmark every claim rests on |
| 7 | **Operator-feedback loop + regime adaptation** (`record_operator_feedback`, `adapt_to_new_regime`, `rebaseline_phase`, JSON checkpointing) | `detector_service.py`, `rolling_baseline.py` | Makes the detector deployable on a live plant without retraining |
| 8 | **Dead-cycle → € / OEE / aluminium-tonnage model** | `dashboard/oee_calculator.py` | Commercial value argument |
| 9 | **Zero-touch deployment and commissioning**: host probing, hardware profiles, plant-agnostic OPC-UA node discovery, sensor-health checks (frozen, noisy, skipped-update, stuck-counter), live calibration with a 70/30 hold-out validation, and a gate that blocks `run` on an unverified plant | `deploy/`, `launch.py` | New; the hold-out gate and skipped-update detection are the distinctive parts |
| 10 | **Pendrive distribution**: offline wheelhouse, per-host environment outside the stick, calibration that travels with the drive and is re-verified (not redone) on each new machine | `launch.py`, `dcto.sh`, `dcto.bat`, `deploy bundle` | Operational know-how rather than novel science |

## Honest caveats

- Items 1–4 are *combinations and tunings* of public methods. That is protectable as a **trade
  secret** (keep the repo private) and possibly as a patent claim on the fused system, but
  novelty of the individual algorithms cannot be claimed. Get patent counsel before relying on it.
- The benchmark is on **simulated** data (`simulator/`). The numbers in the README are not yet
  evidence of performance on a real press; real-plant data would be the most valuable proprietary
  asset and the strongest moat.
- **Privatising the repo does not recall copies already made.** Versions before 24 October 2026 were
  MIT. Anyone who cloned or forked them keeps an irrevocable MIT grant to *those versions*. `LICENSE`
  is now proprietary for everything after. It is a strong all-rights-reserved text, but it is a
  template written without legal review: have counsel fill in the governing law (marked in `LICENSE`),
  confirm the Owner name, and check it holds in your jurisdiction.
- Check the public git history for anything you do not want exposed. Making a repo private does
  not remove it from existing forks or caches.
