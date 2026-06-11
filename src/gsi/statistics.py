"""
Test statistics for paired geo randomization inference.

Each statistic is a function of a 1D residual vector eps_t (length n_pairs)
at a single time point t. The sign-flip randomization framework (Theorem 1)
provides exact finite-sample inference for any statistic.

Statistics:
  t_sgn  — pure sign (count of positive minus negative residuals)
  t_rk   — Wilcoxon signed-rank type (sign × rank of absolute residual)
  t_tr   — studentized trimmed mean (robust to heavy tails)

Key property: All statistics return non-negative scalars (absolute value).
           Ranks and MAD studentizers depend only on |eps| and are thus
           invariant under sign flips — critical for computational caching.
"""

import numpy as np
from scipy import stats as sp_stats


def t_sgn(eps_t: np.ndarray) -> float:
    """Pure sign statistic.

    T_sgn = | sum_i sign(eps_i) |

    sign(0) = 0 per spec convention.

    Args:
        eps_t: 1D residual array, shape (n_pairs,).

    Returns:
        Non-negative scalar statistic.
    """
    return float(np.abs(np.sum(np.sign(eps_t))))


def t_rk(eps_t: np.ndarray, ranks: np.ndarray) -> float:
    """Wilcoxon signed-rank type statistic.

    T_rk = | sum_i sign(eps_i) * R_i |
    where R_i = rank(|eps_i|) / n

    Ranks are precomputed from |eps| and are flip-invariant.

    Args:
        eps_t: 1D residual array, shape (n_pairs,).
        ranks: Normalized ranks of |eps_t|, shape (n_pairs,).

    Returns:
        Non-negative scalar statistic.
    """
    return float(np.abs(np.sum(np.sign(eps_t) * ranks)))


def t_tr(eps_t: np.ndarray, q: float, s_hat: float) -> float:
    """Studentized two-sided trimmed mean statistic.

    T_tr = |trimmed_mean_q(eps)| / s_hat

    where trimmed_mean_q trims ceil(q*n) observations from EACH tail
    (sorted by signed eps value) and s_hat = MAD(|eps|) is computed from
    absolute residuals (flip-invariant).

    If all observations are trimmed (q >= 0.5), returns 0.0.

    Args:
        eps_t: 1D residual array, shape (n_pairs,).
        q: Trimming proportion per tail (e.g. 0.1 trims 10% from each side).
        s_hat: MAD scale estimate (precomputed from |eps| for caching).

    Returns:
        Non-negative scalar statistic.
    """
    n = len(eps_t)
    n_trim = int(np.ceil(q * n))
    if n_trim * 2 >= n:
        return 0.0

    # Sort by signed value — the trim set depends on ordering and changes
    # with each sign flip (NOT flip-invariant).
    sorted_eps = np.sort(eps_t)
    trimmed = sorted_eps[n_trim : n - n_trim]

    if s_hat <= 0.0:
        # Degenerate: all |eps| equal (or zero). If all zero, stat = 0.
        if np.all(eps_t == 0.0):
            return 0.0
        return float(np.abs(np.mean(trimmed)) / 1e-12)

    return float(np.abs(np.mean(trimmed)) / s_hat)


# ── Precomputation helpers (flip-invariant quantities) ──────────────────────


def precompute_ranks(eps: np.ndarray) -> np.ndarray:
    """Precompute normalized ranks of |eps| column-wise.

    Args:
        eps: Residual matrix, shape (n, T).

    Returns:
        Rank matrix, shape (n, T), values in (0, 1].
    """
    n, T = eps.shape
    ranks = np.empty_like(eps, dtype=np.float64)
    abs_eps = np.abs(eps)
    for t in range(T):
        col = abs_eps[:, t]
        # rankdata returns 1..n; normalize to (0, 1]
        ranks[:, t] = sp_stats.rankdata(col) / n
    return ranks


def precompute_mad(eps: np.ndarray) -> np.ndarray:
    """Precompute MAD(|eps|) per column (flip-invariant scale estimate).

    Uses scipy.stats.median_abs_deviation with scale='normal' for
    consistency with normal distribution.

    Args:
        eps: Residual matrix, shape (n, T).

    Returns:
        MAD vector, shape (T,). At least 1e-12 to avoid division by zero.
    """
    _, T = eps.shape
    mad = np.empty(T, dtype=np.float64)
    abs_eps = np.abs(eps)
    for t in range(T):
        mad[t] = sp_stats.median_abs_deviation(abs_eps[:, t], scale='normal', nan_policy='omit')
    mad[mad < 1e-12] = 1e-12
    return mad


def precompute_sign_contrib(eps: np.ndarray) -> np.ndarray:
    """Precompute sign contribution matrix for t_sgn.

    contrib[i,t] = sign(eps[i,t])  (sign(0) = 0)

    This enables fully vectorized flip distribution: for a sign vector sigma,
    stat at time t = |sum_i sigma[i] * contrib[i,t]|.

    Args:
        eps: Residual matrix, shape (n, T).

    Returns:
        Sign matrix, shape (n, T), values in {-1, 0, 1}.
    """
    return np.sign(eps).astype(np.float64)


def precompute_rk_contrib(eps: np.ndarray, ranks: np.ndarray | None = None) -> np.ndarray:
    """Precompute rank-weighted sign contribution matrix for t_rk.

    contrib[i,t] = sign(eps[i,t]) * rank(|eps[i,t]|)

    Args:
        eps: Residual matrix, shape (n, T).
        ranks: Precomputed ranks. If None, computed from |eps|.

    Returns:
        Contribution matrix, shape (n, T).
    """
    if ranks is None:
        ranks = precompute_ranks(eps)
    return (np.sign(eps) * ranks).astype(np.float64)
