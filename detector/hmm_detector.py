"""Tier 3: Hidden Markov Model (HMM) for latent press health state classification."""
from typing import List, Optional
import numpy as np
from hmmlearn import hmm

class HMMDetector:
    """
    Gaussian Hidden Markov Model that decodes the latent operating condition:
    State 0 = Nominal Operating State
    State 1 = Creeping Wear / Degradation State (mild positive shift)
    State 2 = Micro-Stall / Interlock Hesitation State (severe positive shift)
    """

    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self._is_fitted = False

    def initialize_with_priors(self, nominal_mean: float, nominal_std: float):
        """Initializes model parameters using domain physics priors before fitting."""
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=50,
            random_state=self.random_state
        )

        # Transition matrix prior: strong self-transition persistence
        self.model.transmat_ = np.array([
            [0.92, 0.05, 0.03],   # State 0 (Nominal) -> stays mostly 0
            [0.08, 0.88, 0.04],   # State 1 (Wear) -> stays mostly 1
            [0.40, 0.10, 0.50]    # State 2 (Stall) -> quickly recovers to 0
        ])

        # Prior emission means: [Nominal, Wear (+20%), Stall (+50%)]
        self.model.means_ = np.array([
            [nominal_mean],
            [nominal_mean * 1.20],
            [nominal_mean * 1.50]
        ])

        # Prior emission covariances for diag mode: shape (n_components, n_features) = (3, 1)
        var = max(0.005, nominal_std ** 2)
        self.model.covars_ = np.array([
            [var],
            [var * 2.0],
            [var * 5.0]
        ])

        self.model.startprob_ = np.array([0.90, 0.08, 0.02])
        self._is_fitted = True

    def fit(self, training_durations: List[float]):
        """Fits Gaussian HMM parameters from training telemetry using Baum-Welch (EM)."""
        if len(training_durations) < 30:
            # Not enough data for stable EM convergence; initialize with priors
            mean_val = float(np.mean(training_durations)) if training_durations else 2.5
            std_val = float(np.std(training_durations)) if training_durations else 0.1
            self.initialize_with_priors(mean_val, std_val)
            return

        X = np.array(training_durations).reshape(-1, 1)
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=self.random_state
        )
        self.model.fit(X)

        # Sort hidden states by ascending mean so State 0 is always lowest mean (Nominal)
        means = self.model.means_.flatten()
        order = np.argsort(means)
        self.model.means_ = self.model.means_[order]
        self.model.covars_ = self.model.covars_[order]
        self.model.transmat_ = self.model.transmat_[order][:, order]
        self.model.startprob_ = self.model.startprob_[order]

        self._is_fitted = True

    def decode_sequence(self, durations: List[float]) -> List[int]:
        """Decodes the most likely latent state sequence using the Viterbi algorithm."""
        if not self._is_fitted or not durations:
            return [0] * len(durations)

        X = np.array(durations).reshape(-1, 1)
        try:
            _, states = self.model.decode(X, algorithm="viterbi")
            return list(states)
        except Exception:
            return [0] * len(durations)

    def predict_latest_state(self, durations: List[float], window_size: int = 20) -> int:
        """Decodes the most likely hidden state for the most recent observation."""
        if not durations:
            return 0
        window = durations[-window_size:]
        states = self.decode_sequence(window)
        return states[-1] if states else 0
