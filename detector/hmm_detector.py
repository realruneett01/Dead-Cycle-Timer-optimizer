"""Tier 3: Hidden Markov Model (HMM) for latent press health state classification."""
from typing import List, Optional
import numpy as np
from hmmlearn import hmm
from sklearn.cluster import KMeans

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
        self.model.n_features = 1

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

        obs_matrix = np.array(training_durations).reshape(-1, 1)
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=self.random_state,
            init_params="st"
        )
        # hmmlearn's default init gives every state the global variance, which lets
        # EM merge the wear and stall regimes. Seed each state from its k-means
        # cluster instead so EM starts from distinct nominal/wear/stall regimes.
        centers = np.sort(
            KMeans(n_clusters=self.n_states, n_init=10, random_state=self.random_state)
            .fit(obs_matrix).cluster_centers_.ravel()
        )
        labels = np.searchsorted((centers[1:] + centers[:-1]) / 2.0, obs_matrix.ravel())
        self.model.means_ = centers.reshape(-1, 1)
        cluster_vars = [
            float(np.var(obs_matrix[labels == k]))
            if np.any(labels == k) else float(np.var(obs_matrix))
            for k in range(self.n_states)
        ]
        self.model.covars_ = np.array([[max(v, 1e-4)] for v in cluster_vars])
        self.model.fit(obs_matrix)

        # Sort hidden states by ascending mean so State 0 is always lowest mean (Nominal)
        means = self.model.means_.flatten()
        order = np.argsort(means)
        self.model.means_ = self.model.means_[order]
        # The covars_ getter returns full (n, d, d) matrices, but the "diag" setter
        # expects (n, d), so reduce each matrix to its diagonal before reordering.
        diag_covars = np.array([np.diag(c) for c in self.model.covars_])
        self.model.covars_ = diag_covars[order]
        self.model.transmat_ = self.model.transmat_[order][:, order]
        self.model.startprob_ = self.model.startprob_[order]

        self._is_fitted = True

    def export_state(self) -> dict:
        """Serializes fitted HMM parameters for model persistence across restarts."""
        if not self._is_fitted or self.model is None:
            return {"is_fitted": False}
        covs = getattr(self.model, "_covars_", None)
        if covs is None:
            covs = self.model.covars_
        covs_list = covs.tolist() if hasattr(covs, "tolist") else list(covs)
        return {
            "is_fitted": True,
            "n_states": self.n_states,
            "means": self.model.means_.tolist(),
            "covars": covs_list,
            "transmat": self.model.transmat_.tolist(),
            "startprob": self.model.startprob_.tolist()
        }

    def import_state(self, state: dict):
        """Restores HMM parameters from serialized deployment checkpoint."""
        if not state.get("is_fitted", False):
            return
        self.model = hmm.GaussianHMM(
            n_components=state.get("n_states", self.n_states),
            covariance_type="diag",
            random_state=self.random_state
        )
        self.model.n_features = 1
        self.model.means_ = np.array(state["means"])
        covs = np.array(state["covars"])
        if covs.ndim == 3:
            covs = np.array([np.diag(c) for c in covs])
        self.model._covars_ = covs
        self.model.transmat_ = np.array(state["transmat"])
        self.model.startprob_ = np.array(state["startprob"])
        self._is_fitted = True

    def adapt_to_stream(self, durations: List[float]):
        """Online adaptation: refits or updates emission states from recent observations."""
        valid = [float(x) for x in durations if isinstance(x, (int, float)) and np.isfinite(x) and x > 0.0]
        if len(valid) >= 30:
            self.fit(valid)

    def decode_sequence(self, durations: List[float]) -> List[int]:
        """Decodes the most likely latent state sequence using the Viterbi algorithm."""
        if not self._is_fitted or not durations:
            return [0] * len(durations)

        clean = [float(x) if (isinstance(x, (int, float)) and np.isfinite(x) and x > 0.0) else 0.0 for x in durations]
        obs_matrix = np.array(clean).reshape(-1, 1)
        try:
            _, states = self.model.decode(obs_matrix, algorithm="viterbi")
            return list(states)
        except (ValueError, RuntimeError, AttributeError):
            return [0] * len(durations)

    def predict_latest_state(self, durations: List[float], window_size: int = 20) -> int:
        """Decodes the most likely hidden state for the most recent observation."""
        if not durations:
            return 0
        window = durations[-window_size:]
        states = self.decode_sequence(window)
        return states[-1] if states else 0

