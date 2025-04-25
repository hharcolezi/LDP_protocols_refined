from __future__ import annotations

from typing import Iterable, Callable, Sequence, Tuple, List, Any
import numpy as np

Metric = Tuple[float, float]      # (ASR, MSE)
Candidate = Tuple[Any, Metric]    # (Θ,   f(Θ))

__all__ = [
    "pareto_front",
    "select_utopia",
    "select_weighted",
    "select_elbow",
    "select_epsilon_constraint",
    "select_hv_contribution",
    "select_chebyshev", 
    "enumerate_and_select",
]

# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def _dominates(a: Metric, b: Metric) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1]) and (a != b)

def pareto_front(space: Iterable[Any], metric_fn: Callable[[Any], Metric]) -> List[Candidate]:
    """Return list of non-dominated (Θ, f) pairs (simple O(N²) sweep)."""
    frontier: List[Candidate] = []
    for theta in space:
        f_theta = metric_fn(theta)
        if any(_dominates(f, f_theta) for _, f in frontier):
            continue  # dominated → skip
        frontier = [(t, f) for t, f in frontier if not _dominates(f_theta, f)]
        frontier.append((theta, f_theta))
    return frontier

# ---------------------------------------------------------------------------
# Selectors or optimization criteria
# ---------------------------------------------------------------------------

def select_utopia(frontier: List[Candidate]) -> Candidate:
    """
    Choose the Pareto point **closest to the (component-wise) ideal vector**.

    The *ideal* point is the theoretical `0,0`).  
    The solution that minimises the Euclidean distance to that
    ideal is returned.
    """

    vals = np.array([f for _, f in frontier])
    ideal = np.array([0.0, 0.0]) # or vals.min(axis=0) if you want the empirical utopic point
    idx = np.linalg.norm(vals - ideal, axis=1).argmin()
    return frontier[idx]

def select_weighted(frontier: List[Candidate], w: Sequence[float]) -> Candidate:
    """
    **Linear-scalarisation** selector (a.k.a. weighted-sum).

    Minimises  ``Σ wᵢ · fᵢ``  over the Pareto frontier, where  
    *f* = (ASR, MSE) and *w* is a user-supplied weight vector.
    """
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    scores = [w @ f for _, f in frontier]
    return frontier[int(np.argmin(scores))]

def select_elbow(frontier: List[Candidate]) -> Candidate:
    """
    **Elbow / knee-point** selector - pick the solution that maximises
    curvature along the ordered frontier.
    """
    if len(frontier) <= 2:
        return select_utopia(frontier)
    
    # sort by ASR (first objective)
    ordered = sorted(frontier, key=lambda x: x[1][0]) 
    p0, pN = np.array(ordered[0][1]), np.array(ordered[-1][1])
    chord = pN - p0
    norm = np.linalg.norm(chord)
    if norm == 0:
        return ordered[0]
    def dist(pt):
        return np.abs(np.cross(chord, np.array(pt) - p0)) / norm
    idx = max(range(len(ordered)), key=lambda i: dist(ordered[i][1]))
    return ordered[idx]

def select_epsilon_constraint(frontier: List[Candidate],
                              eps_asr: float = 0.1) -> Candidate:
    """
    Keep only solutions with  ASR ≤ ϵ  and choose the one with minimum MSE.
    Falls back to utopia if no point satisfies the constraint.
    """
    feasible = [(θ, f) for θ, f in frontier if f[0] <= eps_asr]  # f[0] = ASR
    if feasible:
        return min(feasible, key=lambda c: c[1][1])              # minimise MSE
    # infeasible: default to utopia
    return select_utopia(frontier)

def select_hv_contribution(frontier: List[Candidate],
                           ref_point: Tuple[float, float] = (1.0, 1.0)) -> Candidate:
    """
    Pick the Pareto point with the **largest hyper-volume contribution**
    (2 -D case: axis-aligned rectangle to the reference point).

    Suitable when you want the single design that improves overall
    diversity/coverage of the Pareto set the most.
    """
    rx, ry = ref_point
    def hv(f):
        dx = max(0.0, rx - f[0])   # ASR distance
        dy = max(0.0, ry - f[1])   # MSE distance
        return dx * dy
    return max(frontier, key=lambda c: hv(c[1]))

def select_chebyshev(frontier: List[Candidate],
                    w: Sequence[float] = (0.5, 0.5),
                    rho: float = 1e-6) -> Candidate:
    """
    Augmented weighted Tchebycheff selector.

    Minimises  max_i w_i·f_i  +  ρ·Σ w_i·f_i
    where f = (ASR, MSE). 
    """
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    best, best_val = None, np.inf
    for θ, f in frontier:
        tche = max(w * f) + rho * (w @ f)
        if tche < best_val:
            best, best_val = (θ, f), tche
    return best

# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------

def enumerate_and_select(
    space: Iterable[Any],
    metric_fn: Callable[[Any], Metric],
    *,
    selector: str = "utopia",
    weights: Sequence[float] | None = None,
    eps_asr: float = 0.1,
    ref_point: Tuple[float, float] = (1.0, 1.0),
    rho: float = 1e-6,
) -> Tuple[Any, Metric, List[Candidate]]:
    """
    Evaluate the metric function over the parameter space and select a solution
    from the Pareto frontier using a chosen multi-objective  optimization strategy.

    Parameters
    ----------
    space : iterable
        The parameter space (e.g., list of candidate values θ).
    metric_fn : callable
        Function that maps each θ to a tuple (ASR, MSE).
    selector : str, default "weighted"
        Strategy to pick a point on the Pareto frontier.
        Must be one of:
        "elbow", "utopia", "weighted", "epsilon_constraint", "hypervolume", "chebyshev".
    weights : sequence of float, optional
        Required for "weighted" and "chebyshev" selectors.
    eps_asr : float, default 0.1
        Used for "epsilon_constraint" selector: maximum ASR allowed.
    ref_point : tuple(float, float), default (1.0, 1.0)
        Used for "hypervolume" selector: defines the reference point.
    rho : float, default 1e-6
        Augmentation parameter for "chebyshev" selector.

    Returns
    -------
    theta_star : any
        Selected parameter θ.
    f_star : tuple(float, float)
        Associated metric (ASR, MSE).
    frontier : list of (θ, (ASR, MSE))
        Full Pareto frontier.
    """
    frontier = pareto_front(space, metric_fn)
    selector = selector.lower()

    if selector == "elbow":
        theta_star, f_star = select_elbow(frontier)
    elif selector == "utopia":
        theta_star, f_star = select_utopia(frontier)
    elif selector == "weighted":
        if weights is None:
            raise ValueError("weights required for weighted selector")
        theta_star, f_star = select_weighted(frontier, weights)
    elif selector == "epsilon_constraint":
        theta_star, f_star = select_epsilon_constraint(frontier, eps_asr=eps_asr)
    elif selector == "hypervolume":
        theta_star, f_star = select_hv_contribution(frontier, ref_point=ref_point)
    elif selector == "chebyshev":
        if weights is None:
            raise ValueError("weights required for chebyshev selector")
        theta_star, f_star = select_chebyshev(frontier, w=weights, rho=rho)
    else:
        raise ValueError(
            "selector must be one of: 'elbow', 'utopia', 'weighted', 'epsilon_constraint','hypervolume', 'chebyshev'"
        )

    return theta_star, f_star, frontier