"""
Time-uniform confidence bands — Theorem 2.

Implements the max-of-weighted-statistics test that provides finite-sample
exact inference valid at any stopping time. The test statistic is:

  M(theta0) = max_{1 <= t <= T_max} omega_t * T_t(theta0)

where T_t is a sign-flip randomization statistic at time t and omega_t are
pre-registered weights. Under H0(theta0), the distribution of M is exactly
simulated by sign-flip randomization (Lemma 1 + Corollary 1.1).

Key guarantee (Theorem 2):
  P(exists t: omega_t * T_t(theta0) > c_alpha) <= alpha

for any data-dependent stopping rule tau <= T_max.
"""

import numpy as np
from numpy.random import Generator
from . import ObsPanel, BandResult
from .residuals import residual_matrix
from .statistics import (
    t_sgn, t_rk, t_tr,
    precompute_ranks, precompute_mad,
)
from .randomization import flip_distribution


# Registry of known statistics with their precomputation needs
_STAT_CONFIG = {
    "sgn": {"fn": t_sgn, "name": "sgn"},
    "rk": {"fn": t_rk, "name": "rk"},
    "tr": {"fn": t_tr, "name": "tr"},
}


def uniform_band_test(
    obs: ObsPanel,
    theta0: float,
    stat: str,
    omega: np.ndarray,
    B: int,
    alpha: float,
    rng: Generator,
) -> BandResult:
    """Test H0(theta0) using the time-uniform max-of-weighted-statistics.

    Args:
        obs: Realized observed panel.
        theta0: Hypothesized constant iROAS value to test.
        stat: Statistic name — "sgn", "rk", or "tr".
        omega: Pre-registered weight vector, shape (T_max,). Non-negative.
        B: Number of randomization draws (larger = more precise p-value).
        alpha: Significance level.
        rng: Random generator.

    Returns:
        BandResult with rejection decision, p-value, critical value.

    Raises:
        ValueError: If stat is unknown or omega shape mismatch.
    """
    T_max = obs.panel.T_max
    omega = np.asarray(omega, dtype=np.float64)

    if omega.shape != (T_max,):
        raise ValueError(f"omega expected shape ({T_max},), got {omega.shape}")
    if stat not in _STAT_CONFIG:
        raise ValueError(f"Unknown stat: {stat!r}. Valid: {list(_STAT_CONFIG.keys())}")

    stat_cfg = _STAT_CONFIG[stat]

    # ── Step 1: Residual matrix ──────────────────────────────────────────
    eps = residual_matrix(obs, theta0)  # (n, T_max)

    # ── Step 2: Precompute flip-invariant quantities ─────────────────────
    n, T = eps.shape
    if stat == "rk":
        ranks_t = precompute_ranks(eps)  # (n, T)
    else:
        ranks_t = None

    if stat == "tr":
        mad_t = precompute_mad(eps)  # (T,)
    else:
        mad_t = None

    # ── Step 3: Observed M statistic ─────────────────────────────────────
    M_obs = _compute_M_obs(eps, omega, stat, ranks_t, mad_t)

    # ── Step 4: Null distribution via sign-flip randomization ────────────
    M_null = flip_distribution(eps, stat, omega, B, rng)  # (B,)

    # ── Step 5: p-value with +1 correction ───────────────────────────────
    p_value = (1.0 + np.sum(M_null >= M_obs)) / (1.0 + B)

    # ── Step 6: Critical value ───────────────────────────────────────────
    # ceil((B+1)(1 - alpha)) order statistic
    # Using method='higher' gives the smallest value such that
    # at most alpha fraction are >= it (conservative).
    c_alpha = float(np.quantile(M_null, 1.0 - alpha, method='higher'))

    return BandResult(
        reject=(p_value <= alpha),
        p_value=p_value,
        c_alpha=c_alpha,
        M_obs=M_obs,
        M_null=M_null,
        theta0=theta0,
        alpha=alpha,
    )


def _compute_M_obs(
    eps: np.ndarray,
    omega: np.ndarray,
    stat: str,
    ranks_t: np.ndarray | None,
    mad_t: np.ndarray | None,
) -> float:
    """Compute M_obs = max_t omega[t] * T_t(eps[:, t]) for the named statistic."""
    n, T = eps.shape
    t_vals = np.empty(T, dtype=np.float64)

    for t in range(T):
        eps_t = eps[:, t]
        if stat == "sgn":
            t_vals[t] = t_sgn(eps_t)
        elif stat == "rk":
            t_vals[t] = t_rk(eps_t, ranks_t[:, t])
        elif stat == "tr":
            t_vals[t] = t_tr(eps_t, q=0.1, s_hat=mad_t[t])  # q=0.1 per spec default
        else:
            raise ValueError(f"Unknown stat: {stat!r}")

    return float(np.max(t_vals * omega))
