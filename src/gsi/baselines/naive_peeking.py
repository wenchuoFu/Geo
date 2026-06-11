"""
Naive peeking baseline: daily t-test with early stopping.

This demonstrates the severe Type I error inflation that occurs when
a fixed-sample t-test is used with continuous monitoring. This is the
baseline that Theorem 2 (time-uniform bands) is designed to replace.

Expected behavior: under H0, the naive approach rejects at rate 30%+
instead of the nominal 5%, because it ignores the multiple looks.

The test:
  - At each day t, compute the ratio r_i = dY_i(t) / dS_i(t) for each pair.
  - Run a one-sample t-test: H0: mean(r_i) = 0.
  - If p < 0.05 at any t, stop and reject.
"""

import numpy as np
from scipy import stats as sp_stats


def naive_peeking_test(
    dY: np.ndarray,
    dS: np.ndarray,
    alpha: float = 0.05,
    min_t: int = 3,
) -> dict:
    """Run the naive peeking t-test on daily ratio estimates.

    Args:
        dY: Cumulative response differences, shape (n_pairs, T_max).
        dS: Cumulative spend differences, shape (n_pairs, T_max).
        alpha: Nominal significance level (ignored by stopping rule — always 0.05).
        min_t: Minimum number of days before first look.

    Returns:
        dict with:
          - reject: Whether H0 was rejected at any t.
          - first_crossing: First day t where p < 0.05, or None.
          - p_values: Array of p-values for each day t.
          - estimates: Array of mean ratio estimates per day.
    """
    n, T = dY.shape

    p_values = np.full(T, np.nan)
    estimates = np.full(T, np.nan)

    for t in range(min_t - 1, T):
        # Ratio estimates at day t
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.divide(dY[:, t], dS[:, t],
                               out=np.full(n, np.nan),
                               where=np.abs(dS[:, t]) > 1e-10)

        valid = ~np.isnan(ratios) & ~np.isinf(ratios)
        if np.sum(valid) < 3:
            continue

        r_valid = ratios[valid]
        estimates[t] = np.mean(r_valid)

        # One-sample t-test: H0: mean = 0
        t_stat, p_val = sp_stats.ttest_1samp(r_valid, popmean=0.0)
        p_values[t] = p_val

    # ── Rejection decision: reject if any p < 0.05 ───────────────────────
    valid_p = p_values[~np.isnan(p_values)]
    if len(valid_p) > 0 and np.any(valid_p < 0.05):
        first_crossing = int(np.where(p_values < 0.05)[0][0])
        reject = True
    else:
        first_crossing = None
        reject = False

    return {
        "reject": reject,
        "first_crossing": first_crossing,
        "p_values": p_values,
        "estimates": estimates,
    }
