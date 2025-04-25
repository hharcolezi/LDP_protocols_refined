from __future__ import annotations
from typing import Iterable, Tuple, Callable
import warnings

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.stats import binom

# silence harmless scipy warnings in the ASR loop
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats")

from ldp_protocols.optimizer import (
    pareto_front,
    select_utopia,
    select_elbow,
    select_weighted,
    select_epsilon_constraint,
    select_hv_contribution,
    select_chebyshev
)

__all__ = ["AdaptiveUnaryEncoding"]

@jit(nopython=True)
def ue_obfuscate(input_data: int, k: int, p: float, q: float) -> np.ndarray:
    """
    Obfuscate the input data using the Unary Encoding (UE) protocol with custom p and q parameters.

    Parameters
    ----------
    input_data : int
        The user's true value to be obfuscated. Must be in the range [0, k-1].
    k : int
        The size of the domain (number of possible values). Must be an integer >= 2.
    p : float
        The probability of retaining a '1' in the UE vector for the true value.
    q : float
        The probability of flipping a '0' to '1' in the UE vector.

    Returns
    -------
    np.ndarray
        An obfuscated unary vector of size `k`.

    Raises
    ------
    ValueError
        If `input_data` is not in the range [0, k-1].
    """
    if input_data < 0 or input_data >= k:
        raise ValueError("input_data must be in the range [0, k-1].")

    # Unary encoding
    input_ue_data = np.zeros(k)
    if input_data is not None:
        input_ue_data[input_data] = 1

    # Initializing a zero-vector
    obfuscated_vec = np.zeros(k)

    # UE perturbation function
    for ind in range(k):
        if input_ue_data[ind] != 1:
            rnd = np.random.random()
            if rnd <= q:
                obfuscated_vec[ind] = 1
        else:
            rnd = np.random.random()
            if rnd <= p:
                obfuscated_vec[ind] = 1
    return obfuscated_vec

@jit(nopython=True)
def attack_ue(obfuscated_vec: np.ndarray, k: int) -> int:
        """
        Perform a privacy attack on an obfuscated unary vector.

        This method attempts to infer the true value from the obfuscated vector. If the vector 
        contains no '1' values (all positions are 0), the method returns a random guess 
        within the domain `[0, k-1]`. Otherwise, it randomly selects one of the indices where 
        the vector has a '1'.

        Parameters
        ----------
        obfuscated_vec : np.ndarray
            An obfuscated unary vector of size `k`, generated using the UE mechanism.

        k : int
            Domain size.

        Returns
        -------
        int
            The inferred true value of the input. If no inference is possible (sum of the vector is 0),
            a random value in the range `[0, k-1]` is returned.
        """

        # If the vector contains no '1', make a random guess
        if np.sum(obfuscated_vec) == 0:
            return np.random.randint(k)
        else:
            # Randomly select one of the indices where the value is '1'
            return np.random.choice(np.where(obfuscated_vec == 1)[0])

class AdaptiveUnaryEncoding:
    """Adaptive Unary Encoding (AUE).

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

    p_grid : Iterable[float], optional
        Custom grid of candidate `p` values to evaluate. Defaults to 
        ``np.linspace(0.5, 0.999999, 100)``.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        k: int,
        epsilon: float,
        optimization: str = "weighted",
        weights: Tuple[float, float] | None = (0.5, 0.5),
        eps_asr: float = 0.1,
        ref_point: Tuple[float, float] = (1.0, 1.0),
        rho: float = 1e-6,
        p_grid: Iterable[float] | None = None,
    ) -> None:
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

        # candidate p grid (continuous parameter → sample uniformly)
        self._p_grid = (
            np.linspace(0.5, 0.999999, 100)
            if p_grid is None
            else np.asarray(list(p_grid), dtype=float)
        )

        # Build frontier (ASR,MSE) for every p
        self._frontier = pareto_front(self._p_grid, self.metrics)  # [(p, (ASR, MSE))]

        # Select best p according to the optimization strategy
        if optimization == "elbow":
            self.p, _ = select_elbow(self._frontier)

        elif optimization == "utopia":
            self.p, _ = select_utopia(self._frontier)

        elif optimization == "weighted":
            if weights is None:
                raise ValueError("weights must be provided for weighted selector")
            w = np.asarray(weights, dtype=float)
            if w.shape != (2,) or (w < 0).any() or w.sum() == 0:
                raise ValueError("weights must be two non-negative numbers")
            self.p, _ = select_weighted(self._frontier, w)

        elif optimization == "epsilon_constraint":
            self.p, _ = select_epsilon_constraint(self._frontier, eps_asr=self.eps_asr)

        elif optimization == "hypervolume":
            self.p, _ = select_hv_contribution(self._frontier, ref_point=self.ref_point)

        elif optimization == "chebyshev":
            if weights is None:
                raise ValueError("weights must be provided for chebyshev selector")
            w = np.asarray(weights, dtype=float)
            if w.shape != (2,) or (w < 0).any() or w.sum() == 0:
                raise ValueError("weights must be two non-negative numbers")
            self.p, _ = select_chebyshev(self._frontier, w / w.sum(), rho=self.rho)

        # derive q from ε-LDP constraint RNG
        self.q = self._q_from_p(self.p)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Parameter space & metrics (required by optimizer)
    # ------------------------------------------------------------------
    def _q_from_p(self, p: float) -> float:
        """Given *p*, return the unique *q* satisfying ε-LDP constraint."""
        return p / (np.exp(self.epsilon) * (1.0 - p) + p)

    def param_space(self) -> Iterable[float]:  # for optimiser API
        return self._p_grid

    def metrics(self, p: float) -> Tuple[float, float]:
        q = self._q_from_p(p)          # derive the matching q
        return self.get_asr(p, q), self.get_mse(p, q)

    # Alias so the optimized import works with a generic callable
    __call__: Callable[[float], Tuple[float, float]] = metrics

    # ------------------------------------------------------------------
    # Core protocol operations
    # ------------------------------------------------------------------

    def obfuscate(self, input_data: int) -> np.ndarray:
        """
        Obfuscate the input data using the optimized AUE mechanism.

        Parameters
        ----------
        input_data : int or None
            The user's true input value. Must be in the range [0, k-1], or None if no value is provided.

        Returns
        -------
        np.ndarray
            An obfuscated unary vector of size `k`.
        """

        return ue_obfuscate(input_data, self.k, self.p, self.q)
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Adaptive Unary Encoding (AUE) mechanism.

        This method applies unbiased estimation to the noisy unary vectors (noisy reports) 
        to recover the approximate frequencies of values in the domain.

        Parameters
        ----------
        noisy_reports : list of np.ndarray
            A list of noisy unary vectors collected from users. Each unary vector 
            has size `k`, where `k` is the size of the domain.

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

        n = len(noisy_reports)
        if n == 0:
            raise ValueError("Noisy reports cannot be empty.")

        # Count the occurrences of each value in the noisy reports
        support_counts = sum(noisy_reports)

        # Unbiased frequency estimation
        freq_estimates = (support_counts - n * self.q) / (n * (self.p - self.q))
        
        # Ensure non-negative estimates and normalize
        return np.maximum(freq_estimates, 0) / np.sum(np.maximum(freq_estimates, 0))
    
    def attack(self, obfuscated_vec: np.ndarray) -> int:
        """
        Perform a privacy attack on an obfuscated unary vector.

        Parameters
        ----------
        obfuscated_vec : np.ndarray
            An obfuscated unary vector of size `k`, generated using the UE mechanism.

        Returns
        -------
        int
            The inferred true value of the input. If no inference is possible (sum of the vector is 0),
            a random value in the range `[0, k-1]` is returned.
        """
        
        return attack_ue(obfuscated_vec, self.k)

    def get_mse(self, p: float = None, q: float = None, n: int = 1) -> float:
        """
        Compute the MSE of the AUE mechanism.

        Parameters
        ----------
        p : float, optional
            Probability of retaining a bit as 1. If None, use the optimized `p`.
        q : float, optional
            Probability of flipping a bit from 0 to 1. If None, use the optimized `q`.

        Returns
        -------
        float
            The MSE of the AUE mechanism.
        """
        if p is None or q is None:
            p, q = self.p, self.q

        return q * (1 - q) / (n * (p - q) ** 2)

    def get_asr(self, p: float = None, q: float = None) -> float:
        """
        Compute the Adversarial Success Rate (ASR) of the AUE mechanism.

        Parameters
        ----------
        p : float, optional
            Probability of retaining a bit as 1. If None, use the optimized `p`.
        q : float, optional
            Probability of flipping a bit from 0 to 1. If None, use the optimized `q`.

        Returns
        -------
        float
            The Adversarial Success Rate (ASR) of the AUE mechanism.
        """
        if p is None or q is None:
            p, q = self.p, self.q

        # ASR for Event E0: The original bit is flipped to 0, and all other bits remain 0
        asr_e0 = (1 - p) * (1 - q) ** (self.k - 1) * (1 / self.k)

        # Sum of ASR for all other events Ei (i >= 1): Original bit retained and i-1 other bits flipped
        asr_sum = 0
        for i in range(1, self.k + 1):
            binom_prob = binom.pmf(i - 1, self.k - 1, q)
            asr_sum += p * (1 / i) * binom_prob

        # Total expected ASR
        return asr_e0 + asr_sum
    
    # ------------------------------------------------------------------
    # visualisation
    # ------------------------------------------------------------------
    def plot_tradeoff(self, log_x: bool = False) -> None:
        if not hasattr(self, "_frontier"):
            raise AttributeError("frontier not cached")

        pts = self._frontier
        pts_sorted = sorted(pts, key=lambda p_f: p_f[1][1])  # by MSE

        plt.scatter([f[1][1] for f in pts], [f[1][0] for f in pts],
                    s=18, alpha=0.3, label="Candidates")
        plt.plot([f[1][1] for f in pts_sorted], [f[1][0] for f in pts_sorted],
                 "k--", label="Pareto Frontier")

        star_mse = self.get_mse()
        star_asr = self.get_asr()
        plt.scatter([star_mse], [star_asr], marker="*", s=220, color="k",
                    label=f"Selected p={self.p:.3f}")

        plt.xlabel("MSE")
        plt.ylabel("ASR")
        if log_x:
            plt.xscale("log")
        plt.title(f"ASR-MSE Trade-off (k={self.k}, ε={self.epsilon:.2f})")
        plt.legend()
        plt.grid(True, ls=":", alpha=0.6)
        plt.show()
