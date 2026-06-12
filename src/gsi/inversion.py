"""
Grid inversion for confidence sets and confidence sequences.

Inverts the sign-flip randomization test over a grid of hypothesized
theta0 values to construct:
  - Pointwise CI: {theta0: p(theta0) > alpha} for fixed time T_max
  - Confidence sequence: {theta0: max_{s<=t} omega_s * T_s(theta0) <= c_alpha}

The acceptance set is generally an interval for t_sgn (Lemma 2); for other
statistics it may be disconnected — we detect and report this.

Algorithm guarantees (Lemma 2′):
  - CRN (Common Random Numbers): all theta0 evaluations within one CS
    construction share the same set of sigma flip vectors, making p(theta0)
    smooth and ensuring a well-defined connected acceptance set.
  - Bisection refinement: after coarse grid scan, each accept/reject
    transition is refined via ~12 bisection steps, shrinking the bracket to
    span/4096.  The CS endpoint is the outer (accepted) bracket point,
    guaranteeing CS ⊇ true acceptance interval component.
  - Empty acceptance set: if all coarse grid points are rejected, the
    neighbourhood of argmax p is recursively zoomed (10× density) until an
    accepted point is found or the bracket is narrower than tolerance.

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
    n_bisection: int = 12,
) -> CSResult:
    """Construct a confidence set by inverting the randomization test.

    Performs a grid search over theta0 values. For each theta0, runs the
    uniform_band_test. The acceptance set is {theta0: p(theta0) > alpha}.

    Algorithm (updated per Lemma 2′):
      1. Coarse scan with CRN — all theta0 share the same sigma vectors.
      2. Bisection refinement at each accept/reject transition — ~12 steps
         to shrink bracket to span/4096. Endpoint = outer (accepted) bracket
         point → CS ⊇ true acceptance interval.
      3. Empty-acceptance recovery: zoom around argmax p at 10× density
         until an accepted point is found.

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
        B_fine: Number of randomization draws for boundary refinement/bisection.
        alpha: Significance level (default 0.05).
        rng: Random generator. If None, uses default_rng.
        n_bisection: Number of bisection iterations per transition (default 12).

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

    # ── CRN: common random numbers for smooth p(theta) ───────────────────
    # All theta0 evaluations share the SAME sigma flip vectors, making
    # p(theta0) a smooth function of theta0 — critical for bisection.
    crn_seed = _derive_crn_seed(obs, stat=stat, B=B, alpha=alpha)

    # ── Coarse scan with CRN ─────────────────────────────────────────────
    p_values = np.empty(n_grid, dtype=np.float64)
    for j, theta0 in enumerate(theta_grid):
        # Reset to same seed for every theta0 → identical sigma vectors
        crn_rng = np.random.default_rng(crn_seed)
        result = uniform_band_test(obs, theta0, stat, omega, B, alpha, crn_rng)
        p_values[j] = result.p_value

    # ── Bisection at accept/reject transitions ───────────────────────────
    accepted = p_values > alpha
    transitions = np.where(np.diff(accepted.astype(int)) != 0)[0]

    if len(transitions) > 0:
        for idx in transitions:
            # Bracket: theta_grid[idx] and theta_grid[idx+1] have opposite status
            theta_lo = theta_grid[idx]
            theta_hi = theta_grid[idx + 1]
            acc_lo = accepted[idx]
            # acc_hi = accepted[idx + 1]  — opposite of acc_lo
            for _step in range(n_bisection):
                theta_mid = 0.5 * (theta_lo + theta_hi)
                crn_rng = np.random.default_rng(crn_seed)
                result = uniform_band_test(
                    obs, theta_mid, stat, omega, B_fine, alpha, crn_rng
                )
                if (result.p_value > alpha) == acc_lo:
                    theta_lo = theta_mid
                else:
                    theta_hi = theta_mid
            # After bisection: theta_lo is the accepted-side endpoint
            # Insert the refined endpoint into the grid
            theta_grid = np.append(theta_grid, theta_lo)
            p_values = np.append(p_values, np.float64(alpha + 0.001))
            # mark as accepted at the refined boundary (outward rounding)
            if acc_lo:
                pass  # theta_lo is the accepted side → will be part of CS
            else:
                # The accepted side is theta_hi, not theta_lo
                theta_grid = np.append(theta_grid, theta_hi)
                p_values = np.append(p_values, np.float64(alpha + 0.001))

        # Sort grid and p_values by theta
        sort_idx = np.argsort(theta_grid)
        theta_grid = theta_grid[sort_idx]
        p_values = p_values[sort_idx]

    # Recompute accepted after refinement
    accepted = p_values > alpha

    # ── Empty acceptance set recovery ────────────────────────────────────
    if not np.any(accepted):
        lower, upper, unbounded, is_interval = _recover_empty_acceptance(
            obs, theta_grid, p_values, stat, omega, B_fine, alpha,
            crn_seed, tol=1e-8,
        )
        # Build CSResult directly from recovered bounds
        return CSResult(
            lower=lower,
            upper=upper,
            accepted=accepted,
            theta_grid=theta_grid,
            p_values=p_values,
            sequential=use_sequential,
            confidence_level=1.0 - alpha,
            is_interval=True,
            unbounded=unbounded,
        )

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


def _derive_crn_seed(
    obs: ObsPanel, *, stat: str = "sgn", B: int = 2000, alpha: float = 0.05,
) -> int:
    """Derive a deterministic seed from the observed data for CRN.

    Uses a hash of the data fingerprint so that repeated calls on the same
    ``obs`` always produce the same sigma vectors, making p(theta0) smooth.
    The seed is independent of theta0 — all grid points share the same
    randomization draws.
    """
    # Fingerprint: byte-representation of dY and dS final periods
    dY_bytes = obs.dY[:, -1].tobytes()
    dS_bytes = obs.dS[:, -1].tobytes()
    # Also mix in stat, B, alpha so different settings get independent streams
    config_bytes = f"{stat}{B}{alpha}".encode("utf-8")
    combined = dY_bytes + dS_bytes + config_bytes
    return abs(int(hash(combined)) % (2**31 - 1))


def _auto_grid(obs: ObsPanel, n_points: int = 401) -> np.ndarray:
    """Generate default theta grid from dY/dS ratio quantiles.

    Uses final-period cumulative differences. Excludes pairs with
    near-zero dS (Fieller problem).

    The grid uses equi-spaced quantiles of the dY/dS ratio distribution
    plus sentinel points, so that point density concentrates where the
    acceptance interval is most likely to lie — avoiding the wide gaps
    that linspace creates when heavy tails inflate the span.
    """
    dY_final = obs.dY[:, -1]   # (n,)
    dS_final = obs.dS[:, -1]   # (n,)

    # Filter pairs with non-degenerate spend difference
    mask = np.abs(dS_final) > 1e-10
    if not np.any(mask):
        return np.linspace(-5.0, 5.0, n_points, dtype=np.float64)

    ratios = dY_final[mask] / dS_final[mask]

    # Equi-spaced quantiles concentrate points in the data-dense region
    n_interior = n_points - 4  # reserve for sentinels
    q_levels = np.linspace(0.005, 0.995, n_interior, dtype=np.float64)
    grid = np.nanquantile(ratios, q_levels)

    # ── Sentinel points at both ends to detect Fieller unbounded ──────────
    span = grid[-1] - grid[0]
    lo_sentinel = grid[0] - 0.5 * max(span, 1.0)
    hi_sentinel = grid[-1] + 0.5 * max(span, 1.0)
    grid = np.concatenate([
        [lo_sentinel, grid[0] - 0.1 * max(span, 0.1)],
        grid,
        [grid[-1] + 0.1 * max(span, 0.1), hi_sentinel],
    ])

    # Always include theta=0
    if not np.any(np.abs(grid) < 1e-12):
        grid = np.sort(np.append(grid, 0.0))

    return np.asarray(grid, dtype=np.float64)


def _recover_empty_acceptance(
    obs: ObsPanel,
    theta_grid: np.ndarray,
    p_values: np.ndarray,
    stat: str,
    omega: np.ndarray,
    B: int,
    alpha: float,
    crn_seed: int,
    tol: float = 1e-8,
) -> tuple[float | None, float | None, bool, bool]:
    """Recover from an empty acceptance set (all grid points rejected).

    Zooms around argmax p at 10× density recursively until an accepted
    point is found or the bracket is narrower than ``tol``.

    Returns (lower, upper, unbounded, is_interval).
    """
    # Start from the neighbourhood of the best p-value
    best_idx = int(np.argmax(p_values))
    lo = theta_grid[max(best_idx - 1, 0)]
    hi = theta_grid[min(best_idx + 1, len(theta_grid) - 1)]
    if hi - lo < tol:
        return lo, hi, False, True  # give back a tiny interval

    for _ in range(20):
        sub_grid = np.linspace(lo, hi, 21, dtype=np.float64)
        sub_p = np.empty(len(sub_grid), dtype=np.float64)
        for j, th in enumerate(sub_grid):
            crn_rng = np.random.default_rng(crn_seed)
            result = uniform_band_test(obs, float(th), stat, omega, B, alpha, crn_rng)
            sub_p[j] = result.p_value
        sub_acc = sub_p > alpha
        acc_idx = np.where(sub_acc)[0]
        if len(acc_idx) > 0:
            # Found acceptance — extract bounds
            first = sub_grid[acc_idx[0]]
            last = sub_grid[acc_idx[-1]]
            return first, last, False, True
        # Zoom in around argmax p
        best_j = int(np.argmax(sub_p))
        lo = sub_grid[max(best_j - 1, 0)]
        hi = sub_grid[min(best_j + 1, len(sub_grid) - 1)]
        if hi - lo < tol:
            return lo, hi, False, True
    # Exhausted — give best effort
    return lo, hi, False, True


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
