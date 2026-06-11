"""
Sign-flip randomization engine (ALG-1).

Implements the core randomization distribution for the paired geo design.
Under H0(theta0), the residual sign structure (Lemma 1) implies that
{sigma_i * eps_i(t)} has the same joint distribution as {eps_i(theta0, t)}
for any fixed sign vector sigma ∈ {-1, +1}^n.

The engine computes M^(b) = max_t omega[t] * stat(sigma^(b) * eps[:,t])
for B random sign vectors and returns the full null distribution.

Performance strategy:
  - t_sgn and t_rk: fully vectorized via BLAS (sigma @ contrib)
  - t_tr: per-flip loop (trim set depends on signed ordering per flip)
  - Custom stat_fn: per-flip loop with fallback
"""

import numpy as np
from numpy.random import Generator
from typing import Callable
from .statistics import (
    t_sgn, t_rk, t_tr,
    precompute_ranks, precompute_mad,
    precompute_sign_contrib, precompute_rk_contrib,
)


def flip_distribution(
    eps: np.ndarray,
    stat_fn: Callable,
    omega: np.ndarray,
    B: int,
    rng: Generator,
) -> np.ndarray:
    """Generate B draws from the randomization distribution of M(theta0).

    Each draw b:
      sigma^(b) ~ Rademacher(1/2)^n  iid
      M^(b) = max_{t=0..T-1} omega[t] * stat(sigma^(b) * eps[:, t])

    Args:
        eps: Residual matrix, shape (n, T). Must be eps(theta0) under H0.
        stat_fn: Test statistic function. Must accept (eps_t: 1D array) -> float.
                 Also accepts string keys "sgn", "rk", "tr" for optimized paths.
        omega: Pre-registered weight vector, shape (T,). Must be non-negative.
        B: Number of randomization draws.
        rng: Random generator.

    Returns:
        M_null: shape (B,) — B draws of the max-over-t weighted statistic.

    Notes:
        Ranks and MAD (flip-invariant quantities) are cached outside the loop.
        For t_tr, the trim set depends on signed ordering and recomputed per flip.
    """
    eps = np.asarray(eps, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    n, T = eps.shape

    if omega.shape != (T,):
        raise ValueError(f"omega expected shape ({T},), got {omega.shape}")

    # ── Dispatch to optimized paths for known statistics ──────────────────
    if callable(stat_fn):
        # Check if it's one of our known functions
        stat_name = getattr(stat_fn, '__name__', '')
        if stat_name in ('t_sgn', 't_rk', 't_tr'):
            return _flip_vectorized(eps, stat_name, omega, B, rng)
        # Custom callable — fall back to per-flip loop
        return _flip_generic(eps, stat_fn, omega, B, rng)

    if isinstance(stat_fn, str):
        stat_name = stat_fn
        if stat_name in ('sgn', 'rk', 'tr'):
            return _flip_vectorized(eps, stat_name, omega, B, rng)
        raise ValueError(f"Unknown stat name: {stat_name!r}")

    raise TypeError(f"stat_fn must be callable or str, got {type(stat_fn)}")


def _flip_vectorized(
    eps: np.ndarray, stat_name: str, omega: np.ndarray, B: int, rng: Generator
) -> np.ndarray:
    """Vectorized flip distribution for known statistics."""
    n, T = eps.shape

    if stat_name == 'sgn':
        # contrib[i,t] = sign(eps[i,t])
        contrib = precompute_sign_contrib(eps)  # (n, T)
        # sigma @ contrib gives raw sum(sign), we need abs later
        sigma = rng.choice(np.array([-1, 1], dtype=np.int8), size=(B, n))
        stats_t = sigma.astype(np.float64) @ contrib  # (B, T)
        # stats_t[b,t] = sum_i sigma[b,i] * sign(eps[i,t]) = raw signed sum
        # t_sgn = |raw_sum|
        weighted = np.abs(stats_t) * omega[np.newaxis, :]  # (B, T)
        M_b = np.max(weighted, axis=1)  # (B,)
        return M_b

    elif stat_name == 'rk':
        # contrib[i,t] = sign(eps[i,t]) * rank(|eps[i,t]|)
        ranks = precompute_ranks(eps)  # (n, T), flip-invariant
        contrib = precompute_rk_contrib(eps, ranks)  # (n, T)
        sigma = rng.choice(np.array([-1, 1], dtype=np.int8), size=(B, n))
        stats_t = sigma.astype(np.float64) @ contrib  # (B, T)
        weighted = np.abs(stats_t) * omega[np.newaxis, :]  # (B, T)
        M_b = np.max(weighted, axis=1)  # (B,)
        return M_b

    elif stat_name == 'tr':
        # t_tr: per-flip loop (trim set changes with sign ordering)
        return _flip_tr_vectorized(eps, omega, B, rng, q=0.1)

    else:
        raise ValueError(f"Unknown stat name: {stat_name!r}")


def _flip_tr_vectorized(
    eps: np.ndarray, omega: np.ndarray, B: int, rng: Generator, q: float = 0.1
) -> np.ndarray:
    """Per-flip loop for t_tr with column-batched inner loop.

    The trimmed mean depends on signed ordering (not flip-invariant),
    so we recompute per flip. MAD is precomputed (flip-invariant).

    Args:
        eps: Residual matrix, shape (n, T).
        omega: Weight vector, shape (T,).
        B: Number of randomization draws.
        rng: Random generator.
        q: Trimming proportion per tail (default 0.1).
    """
    n, T = eps.shape
    n_trim = int(np.ceil(q * n))

    # Precompute MAD per column (flip-invariant)
    mad_t = precompute_mad(eps)  # (T,)

    M_b = np.empty(B, dtype=np.float64)
    sigma_all = rng.choice(np.array([-1, 1], dtype=np.int8), size=(B, n))

    if n_trim * 2 >= n:
        M_b.fill(0.0)
        return M_b

    # Pre-sort |eps| doesn't help for signed sort, but we can still
    # batch-process. For moderate B, a pure Python loop over B with
    # vectorized inner column processing is acceptable.
    # We use argsort of the signed values each time.
    for b in range(B):
        sigma = sigma_all[b]  # (n,)
        # Flip residuals: (n, T)
        eps_flipped = sigma[:, np.newaxis] * eps

        # For each column t: sort, trim, mean, abs, divide by mad
        # Vectorized across T:
        eps_sorted = np.sort(eps_flipped, axis=0)  # (n, T)
        trimmed = eps_sorted[n_trim : n - n_trim, :]  # (n-2*n_trim, T)
        means = np.mean(trimmed, axis=0)  # (T,)
        t_vals = np.abs(means) / mad_t  # (T,)
        M_b[b] = np.max(t_vals * omega)

    return M_b


def _flip_generic(
    eps: np.ndarray, stat_fn: Callable, omega: np.ndarray, B: int, rng: Generator
) -> np.ndarray:
    """Per-flip loop for arbitrary stat_fn (no vectorization)."""
    n, T = eps.shape

    M_b = np.empty(B, dtype=np.float64)
    sigma_all = rng.choice(np.array([-1, 1], dtype=np.int8), size=(B, n))

    for b in range(B):
        sigma = sigma_all[b]
        eps_flipped = sigma[:, np.newaxis] * eps
        # Compute stat per column and take weighted max
        t_vals = np.array([stat_fn(eps_flipped[:, t]) for t in range(T)])
        M_b[b] = np.max(t_vals * omega)

    return M_b
