"""
Omega weight design for time-uniform confidence bands (§4.3).

Pre-registered weights omega_t control how the "bandwidth budget" is
allocated across time. Theorem 2 guarantees exact finite-sample validity
for ANY pre-registered omega (satisfying A6: depends only on pre-period data).

Built-in shapes:
  - pocock:  omega_t ≡ 1  (uniform monitoring)
  - obf:     omega_t = sqrt(t / T_max)  (O'Brien-Fleming style, back-loaded)
  - mid-peak: omega_t = exp(-kappa * (t - t*)^2 / T_max^2)  (peak at target window)

When shape=None, a simple grid-search optimizer tunes omega to maximize
approximate local power in a target time window using pre-period estimates
of noise scale and spend accumulation.
"""

import numpy as np
from . import ObsPanel


def design_omega(
    pre_period: ObsPanel | None,
    shape: str | None,
    target_window: tuple[int, int] | None,
    delta_star: float,
    T_max: int,
    kappa: float = 1.0,
    t_star: float | None = None,
) -> np.ndarray:
    """Design the omega weight vector for time-uniform monitoring.

    Args:
        pre_period: Pre-period data for optimization (may be None for built-in shapes).
        shape: Built-in shape name — "pocock", "obf", "mid-peak", or None for optimize.
        target_window: (t_start, t_end) for targeted power, 0-indexed inclusive.
        delta_star: Anticipated effect size under alternative (for optimization).
        T_max: Total number of time periods.
        kappa: Concentration parameter for mid-peak shape.
        t_star: Center of mid-peak window (default: midpoint of target_window).

    Returns:
        Omega vector, shape (T_max,). Non-negative. Normalized so max_t omega_t = 1.

    Raises:
        ValueError: If shape is unknown.
        NotImplementedError: If shape=None and pre_period is None.
    """
    if shape is not None and shape not in ("pocock", "obf", "mid-peak"):
        raise ValueError(
            f"Unknown omega shape: {shape!r}. "
            f"Valid shapes: 'pocock', 'obf', 'mid-peak', or None for optimization."
        )

    if shape is None and pre_period is None:
        raise NotImplementedError(
            "Omega optimization requires pre_period data. "
            "Provide a pre_period ObsPanel or use a built-in shape."
        )

    t = np.arange(T_max, dtype=np.float64)

    if shape == "pocock":
        omega = np.ones(T_max, dtype=np.float64)

    elif shape == "obf":
        # omega_t = sqrt(t / T_max); omega_0 must be > 0
        # Shift: omega_t = sqrt((t + 1) / T_max) to avoid zero at t=0
        omega = np.sqrt((t + 1.0) / T_max)

    elif shape == "mid-peak":
        if t_star is None:
            if target_window is not None:
                t_star = float(target_window[0] + target_window[1]) / 2.0
            else:
                t_star = float(T_max) / 2.0
        omega = np.exp(-kappa * (t - t_star) ** 2 / (T_max ** 2))

    elif shape is None:
        # Optimization path
        omega = _optimize_omega(pre_period, target_window, delta_star, T_max)

    else:
        raise NotImplementedError(f"Shape {shape!r} not implemented.")

    # Normalize: max_t omega_t = 1
    omega_max = np.max(omega)
    if omega_max > 0:
        omega = omega / omega_max

    return omega.astype(np.float64)


def _optimize_omega(
    pre_period: ObsPanel,
    target_window: tuple[int, int] | None,
    delta_star: float,
    T_max: int,
) -> np.ndarray:
    """Grid-search optimization of omega weights.

    Uses pre-period data to estimate:
      s_tilde_t: noise scale path (MAD of dY - theta_hat * dS per t)
      S_tilde_t: spend accumulation path (mean |dS| per t)

    Approximate drift: mu_t(delta) ∝ delta * S_tilde_t / s_tilde_t

    Searches over parametric omega shapes parameterized by a concentration
    parameter, maximizing:
      min_{t in target_window} mu_t(delta_star) * omega_t

    The optimization is low-dimensional (1 parameter) so grid search suffices.

    Args:
        pre_period: Pre-period observed panel.
        target_window: (t_start, t_end) for power targeting.
        delta_star: Anticipated effect size.
        T_max: Total periods.

    Returns:
        Optimized omega vector, shape (T_max,).
    """
    if target_window is None:
        target_window = (0, T_max - 1)

    t_start, t_end = target_window
    t = np.arange(T_max, dtype=np.float64)

    # ── Estimate noise scale and spend accumulation from pre-period ──────
    from .statistics import precompute_mad

    dY = pre_period.dY.astype(np.float64)
    dS = pre_period.dS.astype(np.float64)

    # Crude theta_hat for pre-period: ratio of medians at final period
    t_end_pre = dY.shape[1] - 1
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.divide(dY[:, t_end_pre], dS[:, t_end_pre],
                           out=np.full(dY.shape[0], np.nan),
                           where=np.abs(dS[:, t_end_pre]) > 1e-10)
    theta_hat = np.nanmedian(ratios)
    if np.isnan(theta_hat):
        theta_hat = 0.0

    # Residuals under theta_hat
    eps_pre = dY - theta_hat * dS  # (n, T_max)

    # Noise scale per t (MAD)
    s_tilde = precompute_mad(eps_pre)  # (T,)
    s_tilde = np.maximum(s_tilde, 1e-12)

    # Spend scale per t (mean absolute spend difference)
    S_tilde = np.mean(np.abs(dS), axis=0)  # (T,)
    S_tilde = np.maximum(S_tilde, 1e-12)

    # Approximate drift at delta_star
    mu_t = np.abs(delta_star) * S_tilde / s_tilde  # (T,)

    # ── Grid search over concentration parameter kappa ────────────────────
    # Search omega(t) = exp(-kappa * (t - t_center)^2 / T_max^2)
    # for kappa in grid, maximize min_{t in window} mu_t * omega_t
    t_center = float(t_start + t_end) / 2.0
    best_kappa = 1.0
    best_obj = -np.inf

    for kappa in np.linspace(0.0, 5.0, 51):
        omega_candidate = np.exp(-kappa * (t - t_center) ** 2 / (T_max ** 2))
        # Normalize
        omega_candidate = omega_candidate / np.max(omega_candidate)

        # Objective: min power in window
        window_slice = slice(t_start, t_end + 1)
        obj = np.min(mu_t[window_slice] * omega_candidate[window_slice])

        if obj > best_obj:
            best_obj = obj
            best_kappa = kappa

    omega_opt = np.exp(-best_kappa * (t - t_center) ** 2 / (T_max ** 2))
    omega_opt = omega_opt / np.max(omega_opt)

    return omega_opt.astype(np.float64)
