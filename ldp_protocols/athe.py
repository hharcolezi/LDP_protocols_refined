from __future__ import annotations

from typing import Iterable, Tuple, Callable, Sequence
import warnings

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.special import loggamma

# silence harmless numba / exp overflow warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from ldp_protocols.optimizer import (
    pareto_front,
    select_utopia,
    select_elbow,
    select_weighted,
    select_epsilon_constraint,
    select_hv_contribution,
    select_chebyshev
)

__all__ = ["AdaptiveThresholdingHistogramEncoding"]

@jit(nopython=True)
def he_obfuscate(input_data: int, k: int, epsilon: float) -> np.ndarray:
    """
    Obfuscate the input data using the Histogram Encoding (HE) protocol.

    Parameters
    ----------
    input_data : int
        The user's true value to be obfuscated. Must be in the range [0, k-1].
    k : int
        The size of the domain (number of possible values). Must be an integer >= 2.
    epsilon : float
        The privacy budget for the LDP mechanism. Must be a positive value.

    Returns
    -------
    np.ndarray
        A numpy array of size `k` representing the unary encoded input with added Laplace noise.

    Raises
    ------
    ValueError
        If `input_data` is not in the range [0, k-1].
    """
    if input_data < 0 or input_data >= k:
        raise ValueError("input_data must be in the range [0, k-1].")
    
    # Unary encode the input
    input_ue_data = np.zeros(k)
    input_ue_data[input_data] = 1.0

    # Add Laplace noise
    return input_ue_data + np.random.laplace(loc=0.0, scale=2 / epsilon, size=k)

@jit(nopython=True)
def attack_the(ss_the, k):
    """
    Perform a privacy attack on an obfuscated vector generated using the Thresholding Histogram Encoding (THE) protocol.

    This attack attempts to infer the true input value by selecting indices where the obfuscated values
    exceed the threshold. If no values exceed the threshold, a random guess is made.

    Parameters
    ----------
    ss_the : np.ndarray
        An obfuscated vector generated using THE, which includes noisy Laplace values.
    k : int
        The size of the domain (number of possible values).

    Returns
    -------
    int
        The inferred true value. If no values exceed the threshold, a random value in the range `[0, k-1]` is returned.
    """

    if sum(ss_the) == 0:
        return np.random.randint(k)
    else:
        return np.random.choice(ss_the)

class AdaptiveThresholdingHistogramEncoding:
    """
    Adaptive Thresholding Histogram Encoding (ATHE).

    Parameters
    ----------
    k : int
        Domain size (≥ 2).
    epsilon : float
        Privacy budget (ε > 0).
    optimization : {"elbow", "utopia", "weighted", "epsilon_constraint", "hypervolume", "chebyshev"}, default "weighted"
        Optimization strategy used to select the best value of `p`:
        • ``elbow`` - selects the Pareto point corresponding to the maximum perpendicular distance
          from the line joining the extreme ASR/MSE trade-off points. Suitable for detecting "knee" points.\\
        • ``utopia`` - selects the point on the Pareto frontier closest (in Euclidean distance)
          to the ideal (ASR → 0, MSE → 0) point.\\
        • ``weighted`` - selects the point that minimises a scalarised weighted sum
          *w₁·ASR + w₂·MSE*, using the user-provided `weights`.\\
        • ``epsilon_constraint`` - selects the point with minimal MSE among those whose ASR is
          below a threshold `eps_asr`. Falls back to utopia if no such point exists.\\
        • ``hypervolume`` - selects the point contributing the largest hypervolume (coverage) with
          respect to a reference point. Promotes diversity and coverage.\\
        • ``chebyshev`` - selects the point that minimises the augmented weighted Chebyshev norm:
          *max(w₁·ASR, w₂·MSE) + ρ·(w₁·ASR + w₂·MSE)*. Useful for capturing edge cases.

    weights : tuple(float, float), optional
        Weight vector used when `optimization` is "weighted" or "chebyshev".
        Must be two non-negative numbers; they are normalized internally.

    eps_asr : float, optional
        Maximum allowed ASR when `optimization` is "epsilon_constraint". Default is 0.1.

    ref_point : tuple(float, float), optional
        Reference point for hypervolume computation when `optimization` is "hypervolume".
        Default is (1.0, 1.0).

    rho : float, optional
        Augmentation term used in "chebyshev" optimization. Default is 1e-6.
    threshold_grid : Iterable[float], optional
        Custom grid of candidate threshold values; defaults to
        ``np.linspace(0.5, 1.0, 100)``.
    """
    def __init__(
        self,
        k: int,
        epsilon: float,
        optimization: str = "weighted",
        weights: Tuple[float, float] | None = (0.5, 0.5),
        eps_asr: float = 0.1,
        ref_point: Tuple[float, float] = (1.0, 1.0),
        rho: float = 1e-6,
        threshold_grid: Iterable[float] | None = None,
    ) -> None:
        # basic sanity checks
        if k < 2:
            raise ValueError("k must be ≥ 2")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")
        if optimization not in {"elbow", "utopia", "weighted", "epsilon_constraint", "hypervolume", "chebyshev"}:
            raise ValueError("Optimization must be 'elbow', 'utopia', 'weighted', 'epsilon_constraint', 'hypervolume', or 'chebyshev")

        self.k: int = k
        self.epsilon: float = epsilon
        self.optimization = optimization
        self.eps_asr = eps_asr
        self.ref_point = ref_point
        self.rho = rho

        # candidate threshold grid
        self._threshold_grid = (
            np.linspace(0.5, 1.0, 100) if threshold_grid is None
            else np.asarray(list(threshold_grid), dtype=float)
        )

        # Build Pareto frontier (ASR,MSE) for every threshold
        self._frontier = pareto_front(self._threshold_grid, self.metrics) # [(threshold, (ASR,MSE))]

        # select operating point
        if optimization == "elbow":
            self.threshold, _ = select_elbow(self._frontier)

        elif optimization == "utopia":
            self.threshold, _ = select_utopia(self._frontier)

        elif optimization == "weighted":
            if weights is None:
                raise ValueError("weights must be provided for weighted selector")
            w = np.asarray(weights, dtype=float)
            if w.shape != (2,) or (w < 0).any() or w.sum() == 0:
                raise ValueError("weights must be two non-negative numbers")
            self.threshold, _ = select_weighted(self._frontier, w)

        elif optimization == "epsilon_constraint":
            self.threshold, _ = select_epsilon_constraint(self._frontier, eps_asr=self.eps_asr)

        elif optimization == "hypervolume":
            self.threshold, _ = select_hv_contribution(self._frontier, ref_point=self.ref_point)

        elif optimization == "chebyshev":
            if weights is None:
                raise ValueError("weights must be provided for chebyshev selector")
            w = np.asarray(weights, dtype=float)
            if w.shape != (2,) or (w < 0).any() or w.sum() == 0:
                raise ValueError("weights must be two non-negative numbers")
            self.threshold, _ = select_chebyshev(self._frontier, w / w.sum(), rho=self.rho)

        # pre-compute p, q and RNG
        self.p, self.q = self._pq(self.threshold)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Parameter space & metrics (required by optimizer)
    # ------------------------------------------------------------------
    def param_space(self) -> Iterable[int]:
        """Return iterable of candidate threshold values."""
        return self._threshold_grid
    
    def _pq(self, threshold: float) -> Tuple[float, float]:
        """Return (p,q) for a given threshold."""
        p = 1.0 - 0.5 * np.exp(self.epsilon * (threshold - 1.0) / 2.0)
        q = 0.5 * np.exp(-self.epsilon * threshold / 2.0)
        return p, q

    def metrics(self, threshold: float) -> Tuple[float, float]:
        """Return (ASR, MSE) for candidate threshold."""
        return self.get_asr(threshold), self.get_mse(threshold)
    
    # alias so the optimiser can treat the instance as a callable
    __call__: Callable[[float], Tuple[float, float]] = metrics

    # ------------------------------------------------------------------
    # Core protocol operations
    # ------------------------------------------------------------------

    def obfuscate(self, input_data: int) -> np.ndarray:
        """
        Obfuscate the input data using the Adaptive THE mechanism.

        Parameters
        ----------
        input_data : int
            The user's true input value. Must be in the range [0, k-1].

        Returns
        -------
        np.ndarray
            An array of indices where the obfuscated vector exceeds the threshold.
        """
        
        # Apply thresholding
        return np.where(he_obfuscate(input_data, self.k, self.epsilon) > self.threshold)[0]
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Adaptive Thresholding Histogram Encoding (ATHE) mechanism.

        This method applies unbiased frequency estimation to recover approximate frequencies of values 
        in the domain `[0, k-1]`. The method uses thresholded noisy reports and corrects for the perturbation 
        introduced by the ATHE mechanism using `p` (true value probability) and `q` (false value probability).

        Parameters
        ----------
        noisy_reports : list of int
            A list of noisy reports collected from users. Each report corresponds to a value that exceeded 
            the adaptive threshold after Laplace noise was added in the ATHE mechanism.

        Returns
        -------
        np.ndarray
            An array of estimated frequencies for each value in the domain.
            The output array has size `k` and sums to 1.

        Raises
        ------
        ValueError
            If `noisy_reports` is empty.
        """
        n = len(noisy_reports)  # Number of reports
        if n == 0:
            raise ValueError("Noisy reports cannot be empty.")
        
        # Count the occurrences of each value in the noisy reports
        support_counts = np.zeros(self.k)
        for report in noisy_reports:
            support_counts[report] += 1

        # Unbiased frequency estimation
        freq_estimates = (support_counts - n * self.q) / (n * (self.p - self.q))

        # Ensure non-negative estimates and normalize
        return np.maximum(freq_estimates, 0) / np.sum(np.maximum(freq_estimates, 0))
    
    def attack(self, obfuscated_vec: np.ndarray) -> int:
        """
        Perform a privacy attack on an obfuscated vector generated using the Adaptive THE mechanism.

        Parameters
        ----------
        obfuscated_vec : np.ndarray
            An obfuscated vector of size `k`, generated using the Adaptive THE mechanism.

        Returns
        -------
        int
            The inferred true value of the input. If no values exceed the threshold, a random value 
            in the range `[0, k-1]` is returned.
        """
        
        return attack_the(obfuscated_vec, self.k)

    def get_mse(self, threshold: float = None, n: int = 1) -> float:
        """
        Compute the MSE of the Adaptive THE mechanism.

        Parameters
        ----------
        threshold : float, optional
            The threshold value to be used for the MSE calculation.
            If None, the optimized threshold value will be used.

        Returns
        -------
        float
            The MSE of the Adaptive THE mechanism.
        """
        tresh = threshold if threshold is not None else self.threshold
        return (2 * np.exp(self.epsilon * tresh / 2) - 1) / (n * (1 + np.exp(self.epsilon * (tresh - 0.5)) - 2 * np.exp(self.epsilon * tresh / 2))**2)

    def get_asr(self, threshold: float = None) -> float:
        """
        Compute the Adversarial Success Rate (ASR) of the Adaptive THE mechanism.

        Parameters
        ----------
        threshold : float, optional
            The threshold value to be used for the ASR calculation.
            If None, the optimized threshold value will be used.

        Returns
        -------
        float
            The Adversarial Success Rate (ASR) of the Adaptive THE mechanism.
        """
        tresh = threshold if threshold is not None else self.threshold

        # Dynamically calculate p and q for the current tresh
        p = 1 - 0.5 * np.exp(self.epsilon * (tresh - 1) / 2)
        q = 0.5 * np.exp(-self.epsilon * tresh / 2)

        term1 = (1 - p) * (1 - q) ** (self.k - 1) * (1 / self.k)
        term2 = 0
        for m in range(1, self.k + 1):
            try:
                # Calculate comb(self.k - 1, m - 1) in log space
                log_comb = loggamma(self.k) - loggamma(m) - loggamma(self.k - m + 1)
                comb_value = np.exp(log_comb)

                # If comb_value is too large, skip or approximate
                if np.isinf(comb_value) or comb_value > 1e308:
                    continue

                term2 += (1 / m) * comb_value * p * (q ** (m - 1)) * ((1 - q) ** (self.k - m))
            
            except OverflowError:
                # Skip this value if an OverflowError occurs
                continue

        # Final ASR calculation
        asr = term1 + term2
        if np.isinf(asr) or np.isnan(asr):
            asr = 0  # Handle overflow/numerical issues by setting ASR to a valid fallback 

        return asr

    # ------------------------------------------------------------------
    # visualisation
    # ------------------------------------------------------------------
    def plot_tradeoff(self, log_x: bool = False) -> None:
        """Plot MSE (x) vs ASR (y) and highlight selected threshold."""
        if not hasattr(self, "_frontier"):
            raise AttributeError("frontier not cached")

        pts = self._frontier
        pts_sorted = sorted(pts, key=lambda p_f: p_f[1][1])  # sort by MSE

        plt.scatter([f[1][1] for f in pts], [f[1][0] for f in pts],
                    s=18, alpha=0.3, label="Candidates")
        plt.plot([f[1][1] for f in pts_sorted], [f[1][0] for f in pts_sorted],
                 "k--", label="Pareto Frontier")

        star_mse = self.get_mse()
        star_asr = self.get_asr()
        plt.scatter([star_mse], [star_asr], marker="*", s=220, color="k",
                    label=f"Selected θ={self.threshold:.3f}")

        plt.xlabel("MSE")
        plt.ylabel("ASR")
        if log_x:
            plt.xscale("log")
        plt.title(f"ASR-MSE Trade-off (k={self.k}, ε={self.epsilon:.2f})")
        plt.legend()
        plt.grid(True, ls=":", alpha=0.6)
        plt.show()
