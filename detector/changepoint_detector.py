"""Tier 2: Changepoint detection using the ruptures library (Pelt and Binseg algorithms)."""
from typing import List, Optional, Tuple
import numpy as np
import ruptures as rpt

class ChangepointDetector:
    """
    Detects discrete regime shifts and structural breaks in sequential press phase durations.
    Ideal for confirming creeping mechanical/hydraulic wear sequences or re-tooling shifts.
    """

    def __init__(
        self,
        min_size: int = 5,
        penalty: float = 3.0,
        model: str = "rbf"
    ):
        self.min_size = min_size
        self.penalty = penalty
        self.model = model

    def _run_segmentation(
        self,
        algo,
        signal: np.ndarray,
        pen: float
    ) -> List[int]:
        """Fits a ruptures segmentation model and extracts valid interior changepoint indices."""
        try:
            algo.fit(signal)
            result = algo.predict(pen=pen)
            return [cp for cp in result if cp < len(signal)]
        except (
            ValueError, TypeError, np.linalg.LinAlgError,
            rpt.exceptions.BadSegmentationParameters,
            rpt.exceptions.NotEnoughPoints, RuntimeError
        ):
            return []

    def _fit_pelt(self, signal: np.ndarray, pen: float) -> Optional[List[int]]:
        """Fits Pelt model and returns changepoints, or None on numerical/parameter failure."""
        algo = rpt.Pelt(model=self.model, min_size=self.min_size, jump=1)
        res = self._run_segmentation(algo, signal, pen)
        return res if res else None

    def _fit_binseg(self, signal: np.ndarray, pen: float) -> List[int]:
        """Fallback changepoint estimation using Binseg if Pelt fails."""
        algo = rpt.Binseg(model="l2", min_size=self.min_size)
        return self._run_segmentation(algo, signal, pen)

    def find_changepoints(
        self,
        durations: List[float],
        custom_penalty: Optional[float] = None
    ) -> List[int]:
        """
        Applies the Pelt (Pruned Exact Linear Time) algorithm on a sequence of durations.
        Returns a list of 0-based indices corresponding to detected change boundaries.
        """
        clean = [float(x) for x in durations if isinstance(x, (int, float)) and np.isfinite(x)]
        if len(clean) < (self.min_size * 2) or float(np.var(clean)) < 1e-5:
            return []

        pen = custom_penalty if custom_penalty is not None else self.penalty
        signal = np.array(clean).reshape(-1, 1)

        changepoints = self._fit_pelt(signal, pen)
        if changepoints is not None:
            return changepoints

        return self._fit_binseg(signal, pen)

    def evaluate_drift_regime(
        self,
        durations: List[float],
        lookback_window: int = 25
    ) -> Tuple[bool, Optional[int], float]:
        """
        Checks whether a changepoint occurred within the recent lookback window,
        and whether the mean duration after the changepoint increased (upward drift).
        Returns: (has_upward_shift, changepoint_index, delta_mean)
        """
        if len(durations) < lookback_window:
            return False, None, 0.0

        sub_seq = durations[-lookback_window:]
        cps = self.find_changepoints(sub_seq)

        if not cps:
            return False, None, 0.0

        # Examine the most recent changepoint in the lookback slice
        latest_cp = cps[-1]
        before_slice = sub_seq[:latest_cp]
        after_slice = sub_seq[latest_cp:]

        if len(before_slice) == 0 or len(after_slice) == 0:
            return False, None, 0.0

        mean_before = float(np.mean(before_slice))
        mean_after = float(np.mean(after_slice))
        delta = mean_after - mean_before

        # If after mean is noticeably higher (> 0.15s), an upward drift regime is confirmed
        is_upward_drift = bool(delta > 0.15)
        absolute_cp_idx = len(durations) - lookback_window + latest_cp

        return is_upward_drift, absolute_cp_idx, round(delta, 3)
