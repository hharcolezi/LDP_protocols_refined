from __future__ import annotations

from sys import maxsize
from typing import Iterable, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
import xxhash 

from ldp_protocols.optimizer import (
    pareto_front,
    select_utopia,
    select_elbow,
    select_weighted,
    select_epsilon_constraint,
    select_hv_contribution,
    select_chebyshev
)
__all__ = [
    "AdaptiveLocalHashing",
]

class AdaptiveLocalHashing:
    """
    Adaptive Local Hashing (ALH).

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
    g_grid : Iterable[int], optional
        Custom grid of *g* values to evaluate; defaults to
        ``range(2, max(k, ⌈e^ε⌉) + 1)``.
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
        g_grid: Iterable[int] | None = None,
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

        # Candidate g grid
        if g_grid is None:
            grid_upper = max(k, int(np.ceil(np.exp(epsilon))) + 1)
            if grid_upper <= 2:
                grid_upper = 3  # ensure it's non-empty
            self._g_grid = np.arange(2, grid_upper)
        else:
            self._g_grid = np.array(list(g_grid))


        # Build frontier (ASR,MSE) for every g
        self._frontier = pareto_front(self._g_grid, self.metrics)  # [(g, (ASR,MSE))]

        if optimization == "elbow":
            self.g, _ = select_elbow(self._frontier)

        elif optimization == "utopia":
            self.g, _ = select_utopia(self._frontier)

        elif optimization == "weighted":
            if weights is None:
                raise ValueError("weights must be provided for weighted selector")
            w = np.asarray(weights, dtype=float)
            if w.shape != (2,) or (w < 0).any() or w.sum() == 0:
                raise ValueError("weights must be two non‑negative numbers")
            self.g, _ = select_weighted(self._frontier, w)

        elif optimization == "epsilon_constraint":
            self.g, _ = select_epsilon_constraint(self._frontier, eps_asr=self.eps_asr)

        elif optimization == "hypervolume":
            self.g, _ = select_hv_contribution(self._frontier, ref_point=self.ref_point)

        elif optimization == "chebyshev":
            if weights is None:
                raise ValueError("weights must be provided for chebyshev selector")
            w = np.asarray(weights, dtype=float)
            if w.shape != (2,) or (w < 0).any() or w.sum() == 0:
                raise ValueError("weights must be two non‑negative numbers")
            self.g, _ = select_chebyshev(self._frontier, w / w.sum(), rho=self.rho)

        # pre‑compute p,q  +  per‑instance RNG for reproducibility
        self.p, self.q = self._pq(self.g)
        self._rng      = np.random.default_rng()

    # ------------------------------------------------------------------
    # Parameter space & metrics (required by optimizer)
    # ------------------------------------------------------------------
    def param_space(self) -> Iterable[int]:
        """Return iterable of g candidates."""
        return self._g_grid

    def _pq(self, g: int) -> Tuple[float, float]:
        p = np.exp(self.epsilon) / (np.exp(self.epsilon) + g - 1)
        q = 1.0 / g
        return p, q

    def metrics(self, g: int) -> Tuple[float, float]:
        """Return (ASR, MSE) for a candidate g."""
        return self.get_asr(g), self.get_mse(g)

    # Alias so the optimized import works with a generic callable
    __call__: Callable[[int], Tuple[float, float]] = metrics

    # ------------------------------------------------------------------
    # Core protocol operations
    # ------------------------------------------------------------------
    def obfuscate(self, input_data: int) -> tuple[int, int]:
        """
        Obfuscate the input data using the ALH mechanism.

        Parameters
        ----------
        input_data : int
            The true input value to be obfuscated. Must be in the range [0, k-1].

        Returns
        -------
        tuple[int, int]
            A tuple containing:
                - The sanitized (obfuscated) value (int) within the optimized hash domain size `g`.
                - The random seed (int) used for hashing.

        Raises
        ------
        ValueError
            If `input_data` is not in the range [0, k-1].
        """
        if input_data < 0 or input_data >= self.k:
            raise ValueError("input_data must be in the range [0, k-1].")

        # Generate random seed and hash the user's value
        rnd_seed = self._rng.integers(0, maxsize, dtype=np.int64)
        hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % self.g)

        # GRR-based perturbation
        domain = np.arange(self.g)
        if self._rng.random() < self.p:
            sanitized_value = hashed_input_data
        else:
            sanitized_value = self._rng.choice(domain[domain != hashed_input_data])

        return sanitized_value, rnd_seed
    
    def _hash_table(self, seed: int) -> np.ndarray:
        """
        helper – pre‑compute hashes of every domain value for a given seed
        Return an int64 array h[0..k-1] with
            h[v] = ( xxh32(str(v), seed).intdigest() mod g )
        """
        return np.fromiter(
            (xxhash.xxh32(str(v), seed=seed).intdigest() % self.g
            for v in range(self.k)),
            dtype=np.int64,
            count=self.k,
        )
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Adaptive Local Hashing (ALH) mechanism.

        This method applies unbiased estimation to recover approximate frequencies of values 
        in the domain `[0, k-1]`. The LH mechanism maps input values to a hash domain of size `g`, 
        perturbs the mapped values, and reports the noisy results. The method uses `p` (true value probability) 
        and `q` (false value probability) to correct for this perturbation.

        Parameters
        ----------
        noisy_reports : list of tuple (int, int)
            A list of noisy reports collected from users. Each report is a tuple containing:
            - `value` : The obfuscated hash-mapped value.
            - `seed`  : The random seed used for hashing during the LH mechanism.

        Returns
        -------
        np.ndarray
            An array of estimated frequencies for each value in the domain `[0, k-1]`.
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

        # cache: seed → hash table
        cache: dict[int, np.ndarray] = {}
        
        # Hash-based support counting for LH protocols
        for value, seed in noisy_reports:
            h = cache.setdefault(seed, self._hash_table(seed))
            support_counts += (h == value)

        # Unbiased frequency estimation
        freq_estimates = (support_counts - n * self.q) / (n * (self.p - self.q))
        
        # Ensure non-negative estimates and normalize
        return np.maximum(freq_estimates, 0) / np.sum(np.maximum(freq_estimates, 0))
    
    def attack(self, val_seed):
        """
        Perform a privacy attack on an obfuscated value generated using the Adaptive Local Hashing (ALH) protocol.

        This method attempts to infer the true input value by leveraging the obfuscated hash-mapped value
        and the corresponding random seed used during hashing. The method reconstructs the possible 
        candidate values that could produce the same hash output and randomly selects one of them.

        Parameters
        ----------
        val_seed : tuple (int, int)
            A tuple containing:
            - `obfuscated value` : The hash-mapped value generated during obfuscation.
            - `seed` : The random seed used for hashing.

        Returns
        -------
        int
            The inferred true value of the input. If no valid candidate values are found, a random value 
            within the domain `[0, k-1]` is returned.
        """

        lh_val = val_seed[0]
        rnd_seed = val_seed[1]

        ss_lh = []
        for v in range(self.k):
            if lh_val == (xxhash.xxh32(str(v), seed=rnd_seed).intdigest() % self.g):
                ss_lh.append(v)

        if len(ss_lh) == 0:
            return self._rng.integers(self.k)
        else:
            return self._rng.choice(ss_lh)

    def get_mse(self, g: int = None, n: int = 1) -> float:
        """
        Compute the MSE of the LH mechanism for a given g.

        Parameters
        ----------
        g : int, optional
            Hash domain size. If None, use the optimized value of g.

        Returns
        -------
        float
            The MSE of the LH mechanism.
        """
        if g is None:
            g = self.g
            
        p = np.exp(self.epsilon) / (np.exp(self.epsilon) + g - 1)
        q = 1 / g

        return q * (1 - q) / (n * (p - q) ** 2)

    def get_asr(self, g: int = None) -> float:
        """
        Compute the Adversarial Success Rate (ASR) of the LH mechanism for a given g.

        Parameters
        ----------
        g : int, optional
            Hash domain size. If None, use the optimized value of g.

        Returns
        -------
        float
            The Adversarial Success Rate (ASR).
        """
        if g is None:
            g = self.g

        return np.exp(self.epsilon) / ((np.exp(self.epsilon) + g - 1) * max(self.k / g, 1))
    
    # ------------------------------------------------------------------
    # visualisation
    # ------------------------------------------------------------------
    def plot_tradeoff(self, log_x: bool = False) -> None:
        """Visualise MSE (x-axis) vs ASR (y-axis) and highlight the chosen point.

        Parameters
        ----------
        log_x : bool, default True
            Plot the MSE axis on a log-10 scale.  Set *False* to keep it linear.
        """
        import matplotlib.pyplot as plt  # local import so the class works head-less

        if not hasattr(self, "_frontier"):
            raise AttributeError("Pareto frontier not cached; make sure the object "
                                 "was initialised with a optimization other than 'manual'.")

        frontier = self._frontier  # list[(g, f)] with f = (ASR, MSE)
        # sort for nicer dashed line: ascending by MSE (index 1)
        frontier_sorted = sorted(frontier, key=lambda x: x[1][1])

        # scatter all candidates (could be many)
        plt.scatter(
            [f[1] for _, f in frontier],   # MSE → x
            [f[0] for _, f in frontier],   # ASR → y
            s=18, alpha=0.3, label="Candidates")

        # draw Pareto curve
        plt.plot(
            [f[1] for _, f in frontier_sorted],  # MSE → x
            [f[0] for _, f in frontier_sorted],  # ASR → y
            "k--", label="Pareto Frontier")

        # highlight chosen operating point
        star_mse = self.get_mse(self.g)
        star_asr = self.get_asr(self.g)
        plt.scatter([star_mse], [star_asr], marker="*", s=220,
                    color="k", label="Selected Point")

        plt.xlabel("MSE")
        plt.ylabel("ASR")
        if log_x:
            plt.xscale("log")
        plt.title(f"MSE-ASR Trade-off (k={self.k}, ε={self.epsilon:.2f})")
        plt.legend()
        plt.grid(True, which="both", linestyle=":", linewidth=0.5)
        plt.show()

