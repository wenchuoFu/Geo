"""
Sub-Gaussian / Hoeffding-type confidence sequence.

Standard anytime-valid confidence sequence based on the sub-Gaussian
assumption. The variance proxy is estimated from the pre-period range
of daily increments.

Under heavy-tailed innovations (t2, Pareto), this CS is expected to
either be extremely wide (conservative proxy) or fail coverage
(underestimated proxy). This is a demonstration baseline for Experiment E4.
"""

import numpy as np
from typing import Iterator


def hoeffding_cs(
    daily_increments: np.ndarray,  # (n_pairs, T_max)
    alpha: float = 0.05,
    variance_proxy: float | None = None,
    pre_period_frac: float = 0.3,
) -> np.ndarray:
    """Compute Hoeffding-type confidence sequence for the mean.

    Args:
        daily_increments: Array of daily observations, shape (n_pairs, T_max).
        alpha: Significance level.
        variance_proxy: Known sub-Gaussian variance proxy. If None, estimated
            from the pre-period range.
        pre_period_frac: Fraction of T to use as pre-period for proxy estimation.

    Returns:
        bounds: shape (T, 2) array of [lower, upper] at each time t.

    Notes:
        This CS is only valid when increments are truly sub-Gaussian with
        the specified variance proxy. With heavy-tailed data, it fails.
    """
    n, T = daily_increments.shape

    # ── Estimate variance proxy from pre-period range ────────────────────
    if variance_proxy is None:
        pre_T = max(1, int(T * pre_period_frac))
        pre_data = daily_increments[:, :pre_T]
        # Range-based proxy: sigma^2 <= (range/2)^2 for bounded data
        data_range = np.max(pre_data) - np.min(pre_data)
        variance_proxy = (data_range / 2.0) ** 2
        variance_proxy = max(variance_proxy, 1e-6)

    # ── Running mean ─────────────────────────────────────────────────────
    # Pool across n pairs for each time t
    cross_sectional_mean = np.mean(daily_increments, axis=0)  # (T,)
    cum_mean = np.cumsum(cross_sectional_mean) / np.arange(1, T + 1)

    # ── Hoeffding width ──────────────────────────────────────────────────
    # Standard sub-Gaussian mixture CS width:
    #   width_t = sqrt(2 * sigma^2 * log(2/alpha + log(t)) / (t * n))
    # or simpler fixed-width version:
    #   width_t = sqrt(2 * sigma^2 * log((log(2*t)+1)/alpha) / (t * n))
    t_arr = np.arange(1, T + 1, dtype=np.float64)
    # Use the "stitching" width for anytime validity
    log_term = np.log((np.log(2.0 * t_arr) + 1.0) / alpha)
    width_t = np.sqrt(2.0 * variance_proxy * log_term / (t_arr * n))

    lower = cum_mean - width_t
    upper = cum_mean + width_t

    return np.column_stack([lower, upper])


def hoeffding_cs_stream(
    stream: Iterator[float],
    alpha: float = 0.05,
    variance_proxy: float = 1.0,
) -> Iterator[tuple[float, float]]:
    """Streaming Hoeffding CS — yields bounds after each observation.

    Args:
        stream: Iterator yielding scalar observations.
        alpha: Significance level.
        variance_proxy: Known sub-Gaussian variance proxy (sigma^2).

    Yields:
        (lower, upper) after each observation.
    """
    cum_sum = 0.0
    t = 0

    for x in stream:
        t += 1
        cum_sum += x
        mean_t = cum_sum / t
        log_term = np.log((np.log(2.0 * t) + 1.0) / alpha)
        width = np.sqrt(2.0 * variance_proxy * log_term / t)
        yield (mean_t - width, mean_t + width)
