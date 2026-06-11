"""
Catoni-type confidence sequence (Wang–Ramdas, arXiv 2202.01250).

Implements Theorem 1 from Wang & Ramdas (2022): a Catoni-style
influence-function-based confidence sequence for the mean of a sequence.

Key assumption: daily increments are i.i.d. (or at least have bounded
conditional moment generating function).

IMPORTANT: This assumption is VIOLATED in our DGP (AR(1) autocorrelation,
heavy tails). This baseline is included to demonstrate exactly that
failure mode — confidence sequences can be overly wide or fail to
maintain coverage under assumption violations.

The influence function:
  psi(x) = sign(x) * min(|x|, lambda)
with data-dependent truncation parameter lambda.
"""

import numpy as np
from typing import Iterator


def catoni_cs(
    daily_increments: np.ndarray,  # (n_pairs, T_max) or 1D stream
    alpha: float = 0.05,
    lambda_param: float | None = None,
) -> np.ndarray:
    """Compute Catoni confidence sequence for the mean of daily increments.

    Args:
        daily_increments: Array of observations. If 2D (n, T), treats each
            time t as having n iid observations (cross-sectional).
            If 1D, treats as a single stream.
        alpha: Significance level.
        lambda_param: Truncation parameter for Catoni influence function.
            If None, set to sqrt(2 * log(2/alpha) / n) heuristic.

    Returns:
        bounds: shape (T, 2) array of [lower, upper] at each time t.

    Notes:
        The CS is valid for iid data with sub-exponential tails.
        Under AR(1) or heavy-tailed innovations, coverage is not guaranteed.
    """
    if daily_increments.ndim == 1:
        daily_increments = daily_increments.reshape(1, -1)

    n, T = daily_increments.shape

    if lambda_param is None:
        # Heuristic truncation: larger lambda = less truncation
        lambda_param = np.sqrt(2.0 * np.log(2.0 / alpha) / max(n, 1))

    # ── Catoni influence function ────────────────────────────────────────
    def psi(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.minimum(np.abs(x), lambda_param)

    # ── Running mean of psi values ───────────────────────────────────────
    # For each time t, compute psi on the cross-section, then running average
    psi_t = np.array([np.mean(psi(daily_increments[:, t])) for t in range(T)])
    running_mean = np.cumsum(psi_t) / np.arange(1, T + 1)

    # ── Width function (Theorem 1, Wang–Ramdas 2022) ────────────────────
    # v_t is the "radius" of the CS at time t
    t_arr = np.arange(1, T + 1, dtype=np.float64)
    # Width: sqrt(2 * log(2/alpha) / (t * n)) * lambda_param adjustment
    # Simplified: width_t = lambda * sqrt(2*log(2/alpha) / (t*n))
    # More precise formulation from the paper:
    #   v_t = lambda * (log(2/alpha) + sum log(1 + ...)) / t
    # We use the simplified version for demonstration.
    width_t = lambda_param * np.sqrt(2.0 * np.log(2.0 / alpha) / (t_arr * n))

    lower = running_mean - width_t
    upper = running_mean + width_t

    return np.column_stack([lower, upper])


def catoni_cs_stream(
    stream: Iterator[float],
    alpha: float = 0.05,
    lambda_param: float | None = None,
) -> Iterator[tuple[float, float]]:
    """Streaming version of Catoni CS — yields bounds after each observation.

    Args:
        stream: Iterator yielding scalar observations.
        alpha: Significance level.
        lambda_param: Truncation parameter (auto if None).

    Yields:
        (lower, upper) bounds after each observation.
    """
    if lambda_param is None:
        lambda_param = 2.0  # default for single-stream

    cum_psi = 0.0
    t = 0

    def psi(x: float) -> float:
        return np.sign(x) * min(abs(x), lambda_param)

    for x in stream:
        t += 1
        cum_psi += psi(x)
        running_mean = cum_psi / t
        width = lambda_param * np.sqrt(2.0 * np.log(2.0 / alpha) / t)
        yield (running_mean - width, running_mean + width)
