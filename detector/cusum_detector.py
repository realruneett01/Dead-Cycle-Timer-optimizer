"""Page's Cumulative Sum (CUSUM) sequential detector grounded in Wald's SPRT theory.

References:
- Page, E. S. (1954). "Continuous Inspection Schemes." Biometrika, 41(1/2), 100-115.
- Wald, A. (1945). "Sequential Tests of Statistical Hypotheses." Ann. Math. Statist. 16(2), 117-186.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class CusumResult:
    cycle_id: int
    phase_name: str
    observed_duration: float
    baseline_mu: float
    baseline_sigma: float
    standardized_residual: float
    cumulative_sum: float
    threshold_h: float
    is_alarm: bool
    excess_seconds: float


class PageCusumDetector:
    """
    Page's CUSUM sequential test for online detection of positive duration shifts.
    
    Mathematical Formulation:
    -------------------------
    Given null hypothesis H0: x_n ~ N(mu_0, sigma_0^2) [nominal cycle]
    vs alternative hypothesis H1: x_n ~ N(mu_0 + delta, sigma_0^2) [delayed cycle].

    From Wald's SPRT log-likelihood ratio, the optimal sequential increment is:
        s_n = (delta / sigma_0^2) * ( (x_n - mu_0) - delta / 2 )

    Page's CUSUM accumulates positive evidence and re-arms after reaching zero:
        S_0 = 0
        S_n = max(0, S_{n-1} + s_n)

    Decision boundary h is derived from target error probabilities (alpha, beta):
        h = ln((1 - beta) / alpha)
    
    An alarm is asserted when S_n >= h, after which S_n is reset to zero to continue
    monitoring subsequent press cycles.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 0.05,
        default_delta_sec: float = 0.35,
        min_baseline_samples: int = 15,
        extreme_z_override: float = 4.0
    ):
        """
        Args:
            alpha: Design false alarm probability (Type I error bound).
            beta: Design missed detection probability (Type II error bound).
            default_delta_sec: Minimal clinically significant delay to detect (e.g. 0.35s).
            min_baseline_samples: Minimum observed cycles before asserting CUSUM alarms.
            extreme_z_override: Single-point Z-score threshold for immediate hard stalls.
        """
        self.alpha = alpha
        self.beta = beta
        self.default_delta_sec = default_delta_sec
        self.min_baseline_samples = min_baseline_samples
        self.extreme_z_override = extreme_z_override

        # Wald boundary: h = ln((1 - beta) / alpha)
        self.boundary_h = float(np.log((1.0 - self.beta) / self.alpha))

        # State per phase: phase_name -> cumulative sum S_n
        self.cusum_state: Dict[str, float] = {}

    def reset_phase(self, phase_name: str):
        """Resets CUSUM accumulator for a specific phase."""
        self.cusum_state[phase_name] = 0.0

    def evaluate_observation(
        self,
        cycle_id: int,
        phase_name: str,
        duration: float,
        baseline_mu: float,
        baseline_sigma: float,
        sample_count: int,
        phase_delta: Optional[float] = None
    ) -> CusumResult:
        """
        Evaluates a single phase duration observation against Page's CUSUM test.
        """
        sigma = max(baseline_sigma, 0.02)  # Guard against division by zero
        delta = phase_delta if phase_delta is not None else self.default_delta_sec
        residual = duration - baseline_mu
        z_score = residual / sigma

        # Initialize accumulator for new phase
        if phase_name not in self.cusum_state:
            self.cusum_state[phase_name] = 0.0

        current_s = self.cusum_state[phase_name]

        # Calculate SPRT log-likelihood increment: s_n = (delta / sigma^2) * (residual - delta / 2)
        sprt_increment = (delta / (sigma ** 2)) * (residual - (delta / 2.0))

        # Page's CUSUM recursion: S_n = max(0, S_{n-1} + s_n)
        updated_s = max(0.0, current_s + sprt_increment)

        # Alarm conditions:
        # 1. Warmup gate: do not trigger on early initial samples
        # 2. Sequential boundary crossing: updated_s >= h AND residual > 0.10s
        # 3. Single-cycle extreme override: z_score >= extreme_z_override
        is_warmed_up = sample_count >= self.min_baseline_samples
        crossed_boundary = (updated_s >= self.boundary_h) and (residual >= 0.10)
        extreme_outlier = (z_score >= self.extreme_z_override) and (residual >= 0.20)

        is_alarm = is_warmed_up and (crossed_boundary or extreme_outlier)

        if is_alarm:
            # Re-arm CUSUM accumulator to continue monitoring running press
            self.cusum_state[phase_name] = 0.0
        else:
            self.cusum_state[phase_name] = updated_s

        return CusumResult(
            cycle_id=cycle_id,
            phase_name=phase_name,
            observed_duration=round(duration, 4),
            baseline_mu=round(baseline_mu, 4),
            baseline_sigma=round(sigma, 4),
            standardized_residual=round(z_score, 3),
            cumulative_sum=round(updated_s, 3),
            threshold_h=round(self.boundary_h, 3),
            is_alarm=is_alarm,
            excess_seconds=round(max(0.0, residual), 4)
        )
