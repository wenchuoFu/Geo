"""
Constant-iROAS specification test (§6).

Tests the sharp null of constant iROAS across pairs (A2).
Under H0, the sign of eps_i(theta) should be symmetric and unrelated to
pair characteristics. We stratify pairs by pre-period spend magnitude and
test for sign-balance differences between strata.

The test statistic:
  T_het = |t_sgn_big - t_sgn_small|

is evaluated at theta_hat (estimated iROAS) via a plug-in sign-flip
randomization test. The plug-in introduces mild conservatism.

If the test rejects, the constant-iROAS assumption (A2) is violated.
The user should then interpret CS results as covering a "sign-balance"
estimand in [min_i theta_i, max_i theta_i] rather than a sharp constant.
"""

import numpy as np
from numpy.random import Generator
from . import ObsPanel
from .residuals import residual_matrix
from .statistics import t_sgn


def spec_test(
    obs: ObsPanel,
    theta_hat: float | None = None,
    B: int = 10000,
    rng: Generator | None = None,
) -> float:
    """Test the constant-iROAS assumption (A2).

    Stratifies pairs into "big" and "small" pre-period spend groups and
    tests whether the sign statistic differs between groups at theta_hat.

    Args:
        obs: Realized observed panel.
        theta_hat: Estimated iROAS for plug-in. If None, uses final-period
            ratio of medians.
        B: Number of randomization draws.
        rng: Random generator.

    Returns:
        p-value for the constancy test. Small p suggests violation of A2.

    Notes:
        The plug-in of theta_hat makes the test mildly conservative.
        This is acknowledged in the paper and verified via simulation.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, T = obs.panel.n_pairs, obs.panel.T_max

    # ── Estimate theta_hat if not provided ───────────────────────────────
    if theta_hat is None:
        dY_final = obs.dY[:, -1]
        dS_final = obs.dS[:, -1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.divide(dY_final, dS_final,
                               out=np.full(n, np.nan),
                               where=np.abs(dS_final) > 1e-10)
        theta_hat = float(np.nanmedian(ratios))
        if np.isnan(theta_hat):
            theta_hat = 0.0

    # ── Stratify by pre-period spend magnitude ────────────────────────────
    # Pre-period: t < entry_day for each pair? Or simply first half of T?
    # Use the first third of T as pre-period proxy for spend magnitude.
    pre_cutoff = max(1, T // 3)
    pre_spend = np.mean(np.abs(obs.dS[:, :pre_cutoff]), axis=1)  # (n,)

    median_spend = np.median(pre_spend)
    big_mask = pre_spend >= median_spend
    small_mask = ~big_mask

    n_big = np.sum(big_mask)
    n_small = np.sum(small_mask)

    if n_big == 0 or n_small == 0:
        # All pairs in one group — cannot test
        return 1.0

    # ── Residuals at theta_hat ───────────────────────────────────────────
    eps = residual_matrix(obs, theta_hat)  # (n, T)
    eps_final = eps[:, -1]  # (n,)

    # ── Observed heterogeneity statistic ─────────────────────────────────
    t_big = t_sgn(eps_final[big_mask])
    t_small = t_sgn(eps_final[small_mask])
    T_obs = np.abs(t_big - t_small)

    # ── Sign-flip randomization ──────────────────────────────────────────
    # Under H0, signs are exchangeable. We flip Z (not eps directly, since
    # eps = Z * e). Flip sigma on eps and recompute.
    T_null = np.empty(B, dtype=np.float64)
    sigma_all = rng.choice(np.array([-1, 1], dtype=np.int8), size=(B, n))

    for b in range(B):
        sigma = sigma_all[b]
        eps_flipped_final = sigma * eps_final
        t_big_b = t_sgn(eps_flipped_final[big_mask])
        t_small_b = t_sgn(eps_flipped_final[small_mask])
        T_null[b] = np.abs(t_big_b - t_small_b)

    # ── p-value with +1 correction ───────────────────────────────────────
    p_value = (1.0 + np.sum(T_null >= T_obs)) / (1.0 + B)

    return float(p_value)
