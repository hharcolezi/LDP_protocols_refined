from __future__ import annotations

from typing import Iterable, Tuple, Callable, Sequence, List
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

from ldp_protocols.optimizer import (
    pareto_front,
    select_utopia,
    select_elbow,
    select_weighted,
    select_epsilon_constraint,
    select_hv_contribution,
    select_chebyshev
)

__all__ = ["AdaptiveSubsetSelection"]

@jit(nopython=True)
def ss_obfuscate(input_data: int, k: int, epsilon: float, omega: int) -> np.ndarray:
    """
    Obfuscate the input data using the Subset Selection (SS) protocol.

    Parameters
    ----------
    input_data : int
        The user's true value to be obfuscated. Must be in the range [0, k-1].
    k : int
        The size of the domain (number of possible values). Must be an integer >= 2.
    epsilon : float
        The privacy budget for the LDP mechanism. Must be a positive value.
    omega : int
        The size of the subset used in the SS mechanism.

    Returns
    -------
    np.ndarray
        A sanitized subset of values of size `omega`.

    Raises
    ------
    ValueError
        If `input_data` is not in the range [0, k-1].
    """
    if input_data < 0 or input_data >= k:
        raise ValueError("input_data must be in the range [0, k-1].")

    # Mapping domain size k to the range [0, ..., k-1]
    domain = np.arange(k)

    # SS parameters
    p = omega * np.exp(epsilon) / (omega * np.exp(epsilon) + k - omega)

    # SS perturbation function
    sub_set = np.zeros(omega, dtype='int64')
    if np.random.random() <= p:
        sub_set[0] = int(input_data)
        sub_set[1:] = np.random.choice(domain[domain != input_data], size=omega - 1, replace=False)
        return sub_set
    else:
        return np.random.choice(domain[domain != input_data], size=omega, replace=False)
    
@jit(nopython=True)
def attack_ss(obfuscated_vec: np.ndarray) -> int:
    """
    Perform a privacy attack on an obfuscated subset generated using the Adaptive Subset Selection (ASS) protocol.

    This method attempts to infer the true value by randomly selecting a value from the obfuscated subset.
    Since the true value is included with higher probability in the subset, an adversary can exploit this 
    to make an educated guess.

    Parameters
    ----------
    obfuscated_vec : np.ndarray
        An obfuscated subset of values generated using the SS protocol. 
        The subset contains a fixed number of values selected from the domain.

    Returns
    -------
    int
        The inferred true value of the input. This is selected randomly from the values present 
        in the obfuscated subset.
    """
                
    return int(np.random.choice(obfuscated_vec))

class AdaptiveSubsetSelection:
    """
    Adaptive Sub-set Selection (ASS).

    Parameters
    ----------
    k : int
        Domain size (≥2).
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
    omg_grid : Iterable[int], optional
        Custom grid of candidate ω; defaults to ``range(1, k)``.
    """
    # ------------------------------------------------------------------
    # Construction & initialisation
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
        omg_grid: Iterable[int] | None = None,
    ) -> None:
        if k < 2:
            raise ValueError("k must be >= 2")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("ε must be a positive finite number")
        if optimization not in {"elbow", "utopia", "weighted", "epsilon_constraint", "hypervolume", "chebyshev"}:
            raise ValueError("Optimization must be 'elbow', 'utopia', 'weighted', 'epsilon_constraint', 'hypervolume', or 'chebyshev")

        self.k: int = k
        self.epsilon: float = epsilon
        self.optimization = optimization
        self.eps_asr = eps_asr
        self.ref_point = ref_point
        self.rho = rho

        # Candidate omega grid
        self._omg_grid = (
            np.arange(1, k) if omg_grid is None else np.array(list(omg_grid), dtype=int)
        )

        # Build frontier (ASR,MSE) for every ω
        self._frontier = pareto_front(self._omg_grid, self.metrics)  # [(ω, (ASR,MSE))]

        if optimization == "elbow":
            self.omega, _ = select_elbow(self._frontier)

        elif optimization == "utopia":
            self.omega, _ = select_utopia(self._frontier)

        elif optimization == "weighted":
            if weights is None:
                raise ValueError("weights must be provided for weighted selector")
            w = np.asarray(weights, dtype=float)
            if w.shape != (2,) or (w < 0).any() or w.sum() == 0:
                raise ValueError("weights must be two non-negative numbers")
            self.omega, _ = select_weighted(self._frontier, w)

        elif optimization == "epsilon_constraint":
            self.omega, _ = select_epsilon_constraint(self._frontier, eps_asr=self.eps_asr)

        elif optimization == "hypervolume":
            self.omega, _ = select_hv_contribution(self._frontier, ref_point=self.ref_point)

        elif optimization == "chebyshev":
            if weights is None:
                raise ValueError("weights must be provided for chebyshev selector")
            w = np.asarray(weights, dtype=float)
            if w.shape != (2,) or (w < 0).any() or w.sum() == 0:
                raise ValueError("weights must be two non-negative numbers")
            self.omega, _ = select_chebyshev(self._frontier, w / w.sum(), rho=self.rho)

        # pre-compute p,q constants & RNG
        self.p, self.q = self._pq(self.omega)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Parameter space & metrics (required by optimizer)
    # ------------------------------------------------------------------
    def param_space(self) -> Iterable[int]:
        """Return iterable of ω candidates."""
        return self._omg_grid

    def _pq(self, omega: int) -> tuple[float, float]:
        p = (omega * np.exp(self.epsilon)) / (omega * np.exp(self.epsilon) + self.k - omega)
        q = (omega * np.exp(self.epsilon) * (omega - 1) + (self.k - omega) * omega) / ((self.k - 1) * (omega * np.exp(self.epsilon) + self.k - omega))

        return p, q

    def metrics(self, omega: int) -> Tuple[float, float]:
        """Return (ASR, MSE) for a candidate ω (size of subset)."""
        return self.get_asr(omega), self.get_mse(omega)

    # Alias so the optimized import works with a generic callable
    __call__: Callable[[int], Tuple[float, float]] = metrics  # for optimiser

    # ------------------------------------------------------------------
    # Core protocol operations
    # ------------------------------------------------------------------

    def obfuscate(self, input_data: int) -> np.ndarray:
        """
        Obfuscate the input data using the Adaptive Subset Selection (ASS) mechanism.

        Parameters
        ----------
        input_data : int
            The user's true input value to be obfuscated. Must be in the range [0, k-1].

        Returns
        -------
        np.ndarray
            A sanitized subset of values of size omega.
        """
        return ss_obfuscate(input_data, self.k, self.epsilon, self.omega)
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Adaptive Subset Selection (ASS) mechanism.

        This method applies unbiased estimation to the collected noisy reports to approximate 
        the true frequencies of values in the domain. It uses SS-specific parameters `p` (true value probability)
        and `q` (false value probability) to correct for the randomized responses.

        Parameters
        ----------
        noisy_reports : list of int
            A list of noisy reports collected from users. Each report corresponds to a single obfuscated value
            within the domain `[0, k-1]`, chosen as part of a subset generated by the SS mechanism.

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
        Perform a privacy attack on an obfuscated subset generated using the Adaptive Subset Selection (ASS) protocol.

        Parameters
        ----------
        obfuscated_vec : np.ndarray
            An obfuscated subset of values generated using the SS protocol. 
            The subset contains a fixed number of values selected from the domain.

        Returns
        -------
        int
            The inferred true value of the input. This is selected randomly from the values present 
        in the obfuscated subset.
        """
        
        return attack_ss(obfuscated_vec)

    def get_mse(self, omega: int = None, n: int = 1) -> float:
        """
        Compute the MSE of the Adaptive Subset Selection (ASS) mechanism.

        Parameters
        ----------
        omega : int, optional
            The subset size. If None, the optimized omega value is used.

        Returns
        -------
        float
            The MSE of the ASS mechanism.
        """
        if omega is None:
            omega = self.omega

        # Dynamically calculate p and q for the current omega
        p = (omega * np.exp(self.epsilon)) / (omega * np.exp(self.epsilon) + self.k - omega)
        q = (omega * np.exp(self.epsilon) * (omega - 1) + (self.k - omega) * omega) / ((self.k - 1) * (omega * np.exp(self.epsilon) + self.k - omega))

        return q * (1 - q) / (n * (p - q) ** 2)

    def get_asr(self, omega: int = None) -> float:
        """
        Compute the Adversarial Success Rate (ASR) for the Adaptive Subset Selection (ASS) mechanism.

        Parameters
        ----------
        omega : int, optional
            The subset size. If None, the optimized omega value is used.

        Returns
        -------
        float
            The ASR of the ASS mechanism.
        """
        if omega is None:
            omega = self.omega
            
        return np.exp(self.epsilon) / (omega * np.exp(self.epsilon) + self.k - omega)

    # ------------------------------------------------------------------
    # visualisation
    # ------------------------------------------------------------------
    def plot_tradeoff(self, log_x: bool = False) -> None:
        """Plot MSE (x) vs ASR (y) and mark the chosen ω."""
        if not hasattr(self, "_frontier"):
            raise AttributeError("frontier not cached – object created incorrectly")

        frontier = self._frontier
        frontier_sorted = sorted(frontier, key=lambda t_f: t_f[1][1])  # sort by MSE

        plt.scatter([f[1][1] for f in frontier], [f[1][0] for f in frontier],
                    s=18, alpha=0.3, label="Candidates")
        plt.plot([f[1][1] for f in frontier_sorted], [f[1][0] for f in frontier_sorted],
                 "k--", label="Pareto Frontier")
        star_mse = self.get_mse()
        star_asr = self.get_asr()
        plt.scatter([star_mse], [star_asr], marker="*", s=220, color="k",
                    label=f"Selected ω={self.omega}")

        plt.xlabel("MSE")
        plt.ylabel("ASR")
        if log_x:
            plt.xscale("log")
        plt.title(f"MSE-ASR Trade-off (k={self.k}, ε={self.epsilon:.2f})")
        plt.legend()
        plt.grid(True, ls=":", alpha=0.6)
        plt.show()
