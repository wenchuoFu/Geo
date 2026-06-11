"""
Grid inversion for confidence sets and confidence sequences.

Inverts the sign-flip randomization test over a grid of hypothesized
theta0 values to construct:
  - Pointwise CI: {theta0: p(theta0) > alpha} for fixed time T_max
  - Confidence sequence: {theta0: max_{s<=t} omega_s * T_s(theta0) <= c_alpha}

The acceptance set is generally an interval for t_sgn (Lemma 2); for other
statistics it may be disconnected — we detect and report this.

Edge cases (§7.6):
  - Fieller unbounded: if grid endpoints are accepted, mark unbounded=True
  - Non-interval acceptance: flag is_interval=False, include raw accepted set
  - n < 15: log warning about discrete p-value granularity
"""

import warnings
import numpy as np
from numpy.random import Generator
from . import ObsPanel, BandResult, CSResult
from .bands import uniform_band_test


def confidence_set(
    obs: ObsPanel,
    theta_grid: np.ndarray | None = None,
    *,
    sequential: bool = False,
    omega: np.ndarray | None = None,
    stat: str = "sgn",
    B: int = 2000,
    B_fine: int = 10000,
    alpha: float = 0.05,
    rng: Generator | None = None,
) -> CSResult:
    """Construct a confidence set by inverting the randomization test.

    Performs a grid search over theta0 values. For each theta0, runs the
    uniform_band_test. The acceptance set is {theta0: p(theta0) > alpha}.

    Uses a two-pass strategy:
      1. Coarse scan with B=R randomization draws over full grid.
      2. Refined scan with B=B_fine near boundary points.

    Args:
        obs: Realized observed panel.
        theta_grid: 1D array of theta0 values to search. If None, auto-generated
            from quantiles of dY/dS ratios.
        sequential: If True, use sequential/time-uniform test (Theorem 2).
            If False, use fixed-time test at T_max.
        omega: Weight vector for time-uniform test. Required if sequential=True.
            If sequential=False and omega is provided, still uses time-uniform test.
        stat: Statistic name — "sgn" (default), "rk", or "tr".
        B: Number of randomization draws for coarse scan.
        B_fine: Number of randomization draws for boundary refinement.
        alpha: Significance level (default 0.05).
        rng: Random generator. If None, uses default_rng.

    Returns:
        CSResult with lower/upper bounds, full p-value curve, and diagnostics.

    Notes:
        For t_sgn, the acceptance set is guaranteed to be a single interval
        (Lemma 2). For t_tr, disconnected acceptance sets are rare but possible;
        they are flagged with is_interval=False.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = obs.panel.n_pairs
    if n < 15:
        warnings.warn(
            f"Small sample size (n={n}): p-value granularity is at least "
            f"2^{{-n}} = {2.0**(-n):.2e}. Results may be coarse."
        )

    T_max = obs.panel.T_max

    # ── Auto-generate theta grid if not provided ─────────────────────────
    if theta_grid is None:
        theta_grid = _auto_grid(obs)

    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    n_grid = len(theta_grid)

    # ── Default omega for sequential ─────────────────────────────────────
    if sequential and omega is None:
        omega = np.ones(T_max, dtype=np.float64)  # Pocock-style uniform
    elif omega is None:
        omega = np.ones(T_max, dtype=np.float64)

    omega = np.asarray(omega, dtype=np.float64)
    use_sequential = sequential or (omega is not None)

    # ── Coarse scan ──────────────────────────────────────────────────────
    p_values = np.empty(n_grid, dtype=np.float64)
    for j, theta0 in enumerate(theta_grid):
        result = uniform_band_test(obs, theta0, stat, omega, B, alpha, rng)
        p_values[j] = result.p_value

    # ── Identify boundary region ─────────────────────────────────────────
    accepted = p_values > alpha

    # Find transition points (where accepted status changes)
    transitions = np.where(np.diff(accepted.astype(int)) != 0)[0]

    # ── Fine scan near boundaries ────────────────────────────────────────
    if B_fine > B and len(transitions) > 0:
        # For each transition, refine the two adjacent grid points
        refine_indices = set()
        for idx in transitions:
            refine_indices.add(idx)
            refine_indices.add(idx + 1)
        refine_indices = sorted(refine_indices)

        for idx in refine_indices:
            if 0 <= idx < n_grid:
                theta0 = theta_grid[idx]
                result = uniform_band_test(
                    obs, theta0, stat, omega, B_fine, alpha, rng
                )
                p_values[idx] = result.p_value

    # Recompute accepted after refinement
    accepted = p_values > alpha

    # ── Extract bounds ───────────────────────────────────────────────────
    lower, upper, unbounded, is_interval = _extract_bounds(
        theta_grid, accepted, p_values, alpha
    )

    return CSResult(
        lower=lower,
        upper=upper,
        accepted=accepted,
        theta_grid=theta_grid,
        p_values=p_values,
        sequential=use_sequential,
        confidence_level=1.0 - alpha,
        is_interval=is_interval,
        unbounded=unbounded,
    )


def _auto_grid(obs: ObsPanel) -> np.ndarray:
    """Generate default theta grid from dY/dS ratio quantiles.

    Uses final-period cumulative differences. Excludes pairs with
    near-zero dS (Fieller problem).
    """
    dY_final = obs.dY[:, -1]   # (n,)
    dS_final = obs.dS[:, -1]   # (n,)

    # Filter pairs with non-degenerate spend difference
    mask = np.abs(dS_final) > 1e-10
    if not np.any(mask):
        # All degenerate: return symmetric grid around 0
        return np.linspace(-5.0, 5.0, 201, dtype=np.float64)

    ratios = dY_final[mask] / dS_final[mask]

    q01 = np.nanquantile(ratios, 0.01)
    q99 = np.nanquantile(ratios, 0.99)

    # Expand range slightly
    margin = max((q99 - q01) * 0.2, 0.1)
    lo = q01 - margin
    hi = q99 + margin

    return np.linspace(lo, hi, 201, dtype=np.float64)


def _extract_bounds(
    theta_grid: np.ndarray,
    accepted: np.ndarray,
    p_values: np.ndarray,
    alpha: float,
) -> tuple[float | None, float | None, bool, bool]:
    """Extract confidence bounds from accepted set.

    Returns (lower, upper, unbounded, is_interval).
    """
    accepted_idx = np.where(accepted)[0]

    if len(accepted_idx) == 0:
        # Empty acceptance set — very rare, all rejected
        return None, None, False, True

    # Check if acceptance set is contiguous (interval)
    diffs = np.diff(accepted_idx)
    is_interval = np.all(diffs == 1)

    # Check unbounded
    unbounded = accepted[0] or accepted[-1]

    if len(accepted_idx) == len(theta_grid):
        # All accepted — completely uninformative
        return None, None, True, True

    # Extract lower/upper from the acceptance set
    first_acc = accepted_idx[0]
    last_acc = accepted_idx[-1]

    lower = None if accepted[0] else theta_grid[first_acc]
    upper = None if accepted[-1] else theta_grid[last_acc]

    return lower, upper, unbounded, is_interval
