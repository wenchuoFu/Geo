"""
Staggered-entry betting confidence sequences — Theorem 3.

When geo pairs enter the experiment at staggered times (by a pre-fixed
calendar, A5), each pair carries its own independent randomization coin Z_k.
This enables a martingale-based betting approach where:

  - Each pair settles one "bet" at its evaluation window end.
  - The wealth process W_k = prod_{j<=k} (1 + lambda_j * s_j * g_j)
    is a non-negative martingale under H0(theta0) (Theorem 3(i)).
  - P(sup_k W_k >= 1/alpha) <= alpha (Ville's inequality, Theorem 3(ii)).
  - Inverting over theta0 yields a confidence sequence valid at any
    stopping time (Theorem 3(iii)).

Betting strategy (aGRAPA, §7.4):
  f_k = s_k * g_k
  lambda_k = clip(sum_{j<k} f_j / (1 + sum_{j<k} f_j^2), -0.5, 0.5)
  g_k = rank(|eps_k| in history) / k

Two-sided: run separate wealth processes for +s_k and -s_k, take max.
Each side uses alpha/2 level.
"""

import numpy as np
from numpy.random import Generator
from typing import Iterator, List
from . import PairOutcome, CSSnapshot


def betting_cs(
    stream: Iterator[PairOutcome],
    theta_grid: np.ndarray,
    alpha: float,
    rng: Generator,
) -> Iterator[CSSnapshot]:
    """Run the staggered-entry betting confidence sequence.

    Processes pairs one at a time as they settle. Maintains per-theta0
    wealth processes (two-sided: plus and minus). Yields a CSSnapshot
    after each pair is settled.

    Args:
        stream: Iterator yielding PairOutcome objects. Each contains
            eps = dY - theta0_true * dS at the pair's evaluation window.
            NOTE: The PairOutcome should contain the raw (dY, dS) or eps
            computed at a reference theta. The caller is responsible for
            creating appropriate PairOutcome streams for EACH theta0.
            (See _run_betting_over_grid for the actual grid logic.)
        theta_grid: 1D array of theta0 values to test.
        alpha: Significance level (two-sided, alpha/2 per side).
        rng: Random generator (reserved for future extensions).

    Yields:
        CSSnapshot after each pair settlement.

    Notes:
        The stream is consumed internally. Each PairOutcome is assumed to
        contain the necessary information to compute eps for all theta0.
        In practice, the caller pre-computes a (n_pairs, len(theta_grid))
        array of residuals and streams them row-by-row.
    """
    # Convert theta_grid to array
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    n_theta = len(theta_grid)

    # α/2 per side for two-sided
    alpha_half = alpha / 2.0
    threshold = 1.0 / alpha_half

    # ── Per-theta0 state ─────────────────────────────────────────────────
    wealth_plus = np.ones(n_theta, dtype=np.float64)   # W⁺: uses +s_k
    wealth_minus = np.ones(n_theta, dtype=np.float64)  # W⁻: uses -s_k
    sum_f_plus = np.zeros(n_theta, dtype=np.float64)
    sum_f_sq_plus = np.zeros(n_theta, dtype=np.float64)
    sum_f_minus = np.zeros(n_theta, dtype=np.float64)
    sum_f_sq_minus = np.zeros(n_theta, dtype=np.float64)
    history_abs_eps: List[np.ndarray] = []  # per-theta0, each is (k,) so far
    for _ in range(n_theta):
        history_abs_eps.append(np.array([], dtype=np.float64))
    # Sticky rejection: once crossed, permanently out (Theorem 3(iii) uses
    # max_{j<=k} W_j, so a theta0 that ever crossed can never re-enter CS).
    crossed = np.zeros(n_theta, dtype=bool)

    k = 0  # pair counter

    for outcome in stream:
        k += 1
        # outcome.eps is the residual at a SINGLE theta0 or a vector across grid?
        # Per spec, PairOutcome carries eps for the pair at its eval window.
        # For grid inversion, we need eps for EACH theta0.
        # The caller should pre-compute this; we check the shape.
        eps_val = np.asarray(outcome.eps, dtype=np.float64)

        if eps_val.ndim == 0:
            # Single theta0 case: broadcast to grid?
            # Actually, the caller iterates over pairs; each pair has one eps
            # for each theta0. So eps_val should be shape (n_theta,).
            eps_val = np.full(n_theta, eps_val.item(), dtype=np.float64)

        # ── Per-theta0 update ────────────────────────────────────────────
        for j in range(n_theta):
            eps_j = eps_val[j]

            # Amplitude: rank of |eps_j| in history
            hist_j = history_abs_eps[j]
            hist_j_new = np.append(hist_j, np.abs(eps_j))
            history_abs_eps[j] = hist_j_new
            # rank in [1, k], normalize to (0, 1]
            g_k = float(np.searchsorted(np.sort(hist_j_new), np.abs(eps_j)) + 1) / k

            s_k_plus = np.sign(eps_j)
            s_k_minus = -s_k_plus

            # f_k and lambda_k for plus side
            f_k_p = s_k_plus * g_k
            denom_p = 1.0 + sum_f_sq_plus[j]
            lambda_k_p = np.clip(sum_f_plus[j] / denom_p, -0.5, 0.5)
            wealth_plus[j] *= (1.0 + lambda_k_p * f_k_p)
            sum_f_plus[j] += f_k_p
            sum_f_sq_plus[j] += f_k_p * f_k_p

            # f_k and lambda_k for minus side
            f_k_m = s_k_minus * g_k
            denom_m = 1.0 + sum_f_sq_minus[j]
            lambda_k_m = np.clip(sum_f_minus[j] / denom_m, -0.5, 0.5)
            wealth_minus[j] *= (1.0 + lambda_k_m * f_k_m)
            sum_f_minus[j] += f_k_m
            sum_f_sq_minus[j] += f_k_m * f_k_m

        # ── Two-sided wealth: max(W⁺, W⁻) ────────────────────────────────
        wealth_max = np.maximum(wealth_plus, wealth_minus)

        # ── Sticky rejection (Theorem 3(iii): max_{j≤k} W_j ≥ 1/α) ─────
        crossed |= (wealth_max >= threshold)
        accepted = ~crossed

        # ── Bounds ───────────────────────────────────────────────────────
        accepted_idx = np.where(accepted)[0]
        if len(accepted_idx) == 0:
            lower, upper = None, None
        else:
            lower = float(np.min(theta_grid[accepted_idx]))
            upper = float(np.max(theta_grid[accepted_idx]))

        yield CSSnapshot(
            k=k,
            eval_time=outcome.eval_time,
            lower=lower,
            upper=upper,
            wealth=wealth_max.copy(),
            accepted=accepted,
        )
