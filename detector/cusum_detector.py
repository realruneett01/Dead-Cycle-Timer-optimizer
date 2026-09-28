"""Page's Cumulative Sum (CUSUM) sequential detector grounded in Wald's SPRT theory.

References:
- Page, E. S. (1954). "Continuous Inspection Schemes." Biometrika, 41(1/2), 100-115.
- Wald, A. (1945). "Sequential Tests of Statistical Hypotheses." Ann. Math. Statist. 16(2), 117-186.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class CusumObservationInput:
    """Parameters for evaluating a single observation in Page's CUSUM sequential test."""
    cycle_id: int
    phase_name: str
    duration: float
    baseline_mu: float
    baseline_sigma: float
    sample_count: int
    phase_delta: Optional[float] = None


@dataclass
class CusumResult:
    """Outcome of a single sequential Page's CUSUM evaluation."""
    # pylint: disable=too-many-instance-attributes
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

        # Wald boundary: h = ln((1 - beta) / alpha) with probability domain validation
        safe_alpha = min(max(float(self.alpha), 1e-6), 0.49)
        safe_beta = min(max(float(self.beta), 1e-6), 0.49)
        self.boundary_h = float(np.log((1.0 - safe_beta) / safe_alpha))

        # State per phase: phase_name -> cumulative sum S_n
        self.cusum_state: Dict[str, float] = {}

    def reset_phase(self, phase_name: str):
        """Resets CUSUM accumulator for a specific phase."""
        self.cusum_state[phase_name] = 0.0

    def export_state(self) -> Dict[str, float]:
        """Serializes CUSUM accumulator state for deployment persistence."""
        return {k: round(float(v), 5) for k, v in self.cusum_state.items() if np.isfinite(v)}

    def import_state(self, state: Dict[str, float]):
        """Restores CUSUM accumulators from deployment checkpoint."""
        self.cusum_state = {k: max(0.0, float(v)) for k, v in state.items() if np.isfinite(v)}

    @staticmethod
    def _resolve_obs(observation: Optional[CusumObservationInput], kwargs: dict) -> CusumObservationInput:
        """Normalizes input observation object or keyword arguments."""
        if observation is not None:
            return observation
        return CusumObservationInput(
            cycle_id=kwargs["cycle_id"],
            phase_name=kwargs["phase_name"],
            duration=kwargs["duration"],
            baseline_mu=kwargs["baseline_mu"],
            baseline_sigma=kwargs["baseline_sigma"],
            sample_count=kwargs["sample_count"],
            phase_delta=kwargs.get("phase_delta")
        )

    def _is_alarm(self, obs: CusumObservationInput, updated_s: float, residual: float) -> bool:
        """Determines whether sequential CUSUM or extreme single-cycle criteria are met."""
        if obs.sample_count < self.min_baseline_samples:
            return False
        if (updated_s >= self.boundary_h) and (residual >= 0.10):
            return True
        sigma = float(obs.baseline_sigma) if (np.isfinite(obs.baseline_sigma) and obs.baseline_sigma > 0.02) else 0.02
        return (residual / sigma >= self.extreme_z_override) and (residual >= 0.20)

    def _sanitize_parameters(
        self,
        obs: CusumObservationInput
    ) -> Tuple[float, float, float, float]:
        """Validates and regularizes baseline and duration inputs against non-finite anomalies."""
        raw_sigma = obs.baseline_sigma
        sigma = float(raw_sigma) if (np.isfinite(raw_sigma) and raw_sigma > 0.02) else 0.02

        raw_delta = obs.phase_delta if obs.phase_delta is not None else self.default_delta_sec
        delta = float(raw_delta) if (np.isfinite(raw_delta) and raw_delta > 0.01) else self.default_delta_sec

        raw_mu = obs.baseline_mu
        mu = float(raw_mu) if np.isfinite(raw_mu) else obs.duration
        raw_dur = obs.duration
        duration = float(raw_dur) if np.isfinite(raw_dur) else mu
        return sigma, delta, mu, duration

    @staticmethod
    def _calculate_sprt_update(
        current_s: float,
        delta: float,
        sigma: float,
        residual: float
    ) -> float:
        """Calculates bounded sequential increment and updates running cumulative sum."""
        safe_s = current_s if np.isfinite(current_s) else 0.0
        sprt_increment = (delta / (sigma ** 2)) * (residual - (delta / 2.0))
        sprt_increment = float(np.clip(sprt_increment, -50.0, 50.0))
        return max(0.0, safe_s + sprt_increment)

    def evaluate_observation(
        self,
        observation: Optional[CusumObservationInput] = None,
        **kwargs
    ) -> CusumResult:
        """
        Evaluates a single phase duration observation against Page's CUSUM test.
        """
        obs = self._resolve_obs(observation, kwargs)
        sigma, delta, mu, duration = self._sanitize_parameters(obs)

        residual = duration - mu
        z_score = residual / sigma

        current_s = self.cusum_state.get(obs.phase_name, 0.0)
        updated_s = self._calculate_sprt_update(current_s, delta, sigma, residual)

        is_alarm = self._is_alarm(obs, updated_s, residual)
        self.cusum_state[obs.phase_name] = 0.0 if is_alarm else updated_s

        return CusumResult(
            cycle_id=obs.cycle_id,
            phase_name=obs.phase_name,
            observed_duration=round(duration, 4),
            baseline_mu=round(mu, 4),
            baseline_sigma=round(sigma, 4),
            standardized_residual=round(z_score, 3),
            cumulative_sum=round(updated_s, 3),
            threshold_h=round(self.boundary_h, 3),
            is_alarm=is_alarm,
            excess_seconds=round(max(0.0, residual), 4)
        )

