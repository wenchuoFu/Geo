"""
Data-generating process for paired geo experiments.

Generates synthetic latent paths for response (Y) and spend (S) under the
design-based potential outcomes framework. All latent quantities are fixed
constants; only the treatment assignment Z is random.

Paths are cumulative over time. The DGP respects:
  - Sharp null A2: Y^T_g(t) = Y^C_g(t) + theta * (S^T_g(t) - S^C_g(t))
  - AR(1) autocorrelation on daily increments
  - Configurable tail behaviour for innovations
  - Optional seasonality and heterogeneous effects
"""

import numpy as np
from numpy.random import Generator
from . import GeoPanel, ObsPanel


def make_paths(
    n_pairs: int,
    T_max: int,
    theta: np.ndarray,        # (n_pairs,) true iROAS, constant under sharp null
    tail: str,                # "lognormal" | "pareto" | "t2" | "gaussian"
    rho: float,               # AR(1) autocorrelation, 0 <= rho < 0.95
    seasonality: bool,
    hetero_scale: float,
    entry_days: np.ndarray | None,  # (n_pairs,), None = all enter at t=0
    signal_scale: float = 1.0,      # multiplier on delta_S only (not noise) — calibrates signal-to-noise
    rng: np.random.Generator = None,
) -> GeoPanel:
    """Generate latent potential-outcome paths for a paired geo experiment.

    Args:
        n_pairs: Number of geo pairs.
        T_max: Number of time periods (days).
        theta: True per-pair iROAS, shape (n_pairs,). Constant under sharp null.
        tail: Innovation distribution family.
        rho: AR(1) autocorrelation coefficient for daily increments.
        seasonality: Whether to add sinusoidal seasonal pattern.
        hetero_scale: Standard deviation of pair-level heterogeneity in baseline ROAS.
        entry_days: Day each pair enters (0-indexed). None means all enter at t=0.
        signal_scale: Multiplier on delta_S to calibrate signal-to-noise ratio.
            Only affects the treatment spend increment, not the noise paths.
        rng: Random generator.

    Returns:
        GeoPanel with Y_C, S_C, delta_S latent paths.
    """
    if not (0 <= rho < 0.95):
        raise ValueError(f"rho must be in [0, 0.95), got {rho}")
    if tail not in ("lognormal", "pareto", "t2", "gaussian"):
        raise ValueError(f"Unknown tail: {tail!r}")

    if entry_days is None:
        entry_days = np.zeros(n_pairs, dtype=int)
    else:
        entry_days = np.asarray(entry_days, dtype=int)

    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 0:
        # Scalar — broadcast to all pairs
        theta = np.full(n_pairs, theta.item(), dtype=np.float64)
    elif theta.shape != (n_pairs,):
        raise ValueError(
            f"theta expected shape ({n_pairs},) or scalar, got {theta.shape}"
        )

    # ── 1. Generate daily spend increments ──────────────────────────────────
    # Each pair/geo has independent spend paths.
    # Innovations are drawn from the specified tail distribution.
    innovations = _draw_innovations(n_pairs, 2, T_max, tail, rng)
    # Shape: (n_pairs, 2, T_max) — positive daily spend increments

    # ── 2. Apply AR(1) structure ────────────────────────────────────────────
    # The AR(1) is applied to log-spend-increments for positivity, or directly
    # to the increments if they stay positive.
    s_inc = _apply_ar1(innovations, rho)
    # Shape: (n_pairs, 2, T_max)

    # ── 3. Cumulative spend paths ───────────────────────────────────────────
    S_C = np.cumsum(s_inc, axis=2)
    # Shape: (n_pairs, 2, T_max)

    # ── 4. Treatment spend increment (delta_S) ──────────────────────────────
    # Treatment increases spend by a random factor per pair/geo.
    # delta_S is drawn once per pair/geo as a fraction of final control spend.
    # signal_scale calibrates the signal-to-noise ratio — only affects delta_S.
    spend_lift = 0.05 + 0.10 * rng.random(size=(n_pairs, 2))  # Uniform(0.05, 0.15)
    delta_S = spend_lift[:, :, np.newaxis] * S_C * signal_scale
    # Shape: (n_pairs, 2, T_max)

    # ── 5. Baseline ROAS and control response ───────────────────────────────
    # Each pair has a baseline ROAS around 1.0–2.0, plus pair-level heterogeneity.
    baseline_roas = 1.2 + 0.6 * rng.random(size=(n_pairs, 2))
    if hetero_scale > 0:
        baseline_roas += hetero_scale * rng.normal(size=(n_pairs, 2))
    baseline_roas = np.maximum(baseline_roas, 0.1)  # ensure positivity

    # Response noise: from tail distribution, scaled
    resp_noise_scale = 0.3
    resp_noise = _draw_response_noise(n_pairs, 2, T_max, tail, resp_noise_scale, rng)
    # Apply AR(1) to response noise as well
    resp_noise = _apply_ar1(resp_noise, rho * 0.5)  # milder autocorr for response
    # Cumulative response noise
    resp_noise_cum = np.cumsum(resp_noise, axis=2)

    # Control response: baseline_roas * S_C + noise
    Y_C = baseline_roas[:, :, np.newaxis] * S_C + resp_noise_cum
    # Shape: (n_pairs, 2, T_max)

    # ── 6. Seasonality ──────────────────────────────────────────────────────
    if seasonality:
        # Weekly cycle (period=7), additive on response
        t = np.arange(T_max, dtype=np.float64)
        seasonal = 0.05 * np.sin(2 * np.pi * t / 7.0) + 0.03 * np.cos(2 * np.pi * t / 30.0)
        Y_C += seasonal[np.newaxis, np.newaxis, :]

    return GeoPanel(
        n_pairs=n_pairs,
        T_max=T_max,
        Y_C=Y_C,
        S_C=S_C,
        delta_S=delta_S,
        theta=theta,
        entry_days=entry_days,
    )


def realize(panel: GeoPanel, Z: np.ndarray) -> ObsPanel:
    """Realize treatment assignment Z into observed differences.

    For each pair i:
      - If Z_i == +1: treated geo = 0, control geo = 1
      - If Z_i == -1: treated geo = 1, control geo = 0

    Under sharp null A2:
      Y^T_g = Y^C_g + theta_i * delta_S_g
      S^T_g = S^C_g + delta_S_g

    The observed difference is:
      dY_i(t) = Y_treated(t) - Y_control(t)
      dS_i(t) = S_treated(t) - S_control(t)

    Args:
        panel: GeoPanel with latent paths.
        Z: Assignment vector, shape (n_pairs,). +1 or -1.

    Returns:
        ObsPanel with dY and dS differences.
    """
    Z = np.asarray(Z, dtype=np.float64)
    n = panel.n_pairs

    if Z.shape != (n,):
        raise ValueError(f"Z expected shape ({n},), got {Z.shape}")

    # Build treatment indicator: which geo index is treated for each pair
    # Z_i = +1 -> geo 0 treated; Z_i = -1 -> geo 1 treated
    treated_idx = np.where(Z > 0, 0, 1)       # (n_pairs,)
    control_idx = np.where(Z > 0, 1, 0)       # (n_pairs,)

    # Gather treated and control paths via advanced indexing
    pair_idx = np.arange(n)

    # Spend
    S_treated = panel.S_C[pair_idx, treated_idx, :] + panel.delta_S[pair_idx, treated_idx, :]
    S_control = panel.S_C[pair_idx, control_idx, :]
    dS = S_treated - S_control                     # (n_pairs, T_max)

    # Response: under sharp null, Y_T = Y_C + theta * delta_S
    Y_treated = (
        panel.Y_C[pair_idx, treated_idx, :]
        + panel.theta[:, np.newaxis] * panel.delta_S[pair_idx, treated_idx, :]
    )
    Y_control = panel.Y_C[pair_idx, control_idx, :]
    dY = Y_treated - Y_control                     # (n_pairs, T_max)

    return ObsPanel(panel=panel, Z=Z, dY=dY, dS=dS)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _draw_innovations(
    n_pairs: int, n_geos: int, T_max: int, tail: str, rng: Generator
) -> np.ndarray:
    """Draw daily spend increments (positive) from the specified tail distribution.

    Returns shape (n_pairs, n_geos, T_max).
    """
    size = (n_pairs, n_geos, T_max)
    if tail == "gaussian":
        # Lognormal base (spend must be positive)
        raw = rng.lognormal(mean=0.0, sigma=0.5, size=size)
    elif tail == "lognormal":
        raw = rng.lognormal(mean=0.0, sigma=0.8, size=size)
    elif tail == "pareto":
        # Pareto shape=2.5 (finite mean, infinite variance at low shape)
        raw = rng.pareto(a=2.5, size=size) + 0.1  # shift to avoid zero
    elif tail == "t2":
        # t with df=2: heavy-tailed, infinite variance
        raw = np.abs(rng.standard_t(df=2, size=size)) + 0.1
    else:
        raise ValueError(f"Unknown tail: {tail!r}")
    return raw


def _draw_response_noise(
    n_pairs: int, n_geos: int, T_max: int, tail: str, scale: float, rng: Generator
) -> np.ndarray:
    """Draw response noise innovations. Can be negative.

    Returns shape (n_pairs, n_geos, T_max).
    """
    size = (n_pairs, n_geos, T_max)
    if tail == "gaussian":
        raw = scale * rng.normal(size=size)
    elif tail == "lognormal":
        # Lognormal noise shifted to have mean ~ 0
        raw = scale * (rng.lognormal(mean=0.0, sigma=0.5, size=size) - np.exp(0.125))
    elif tail == "pareto":
        raw = scale * (rng.pareto(a=2.5, size=size) - 1.0 / (2.5 - 1))
    elif tail == "t2":
        raw = scale * rng.standard_t(df=2, size=size)
    else:
        raise ValueError(f"Unknown tail: {tail!r}")
    return raw


def _apply_ar1(innovations: np.ndarray, rho: float) -> np.ndarray:
    """Apply AR(1) structure to daily innovations.

    x_t = rho * x_{t-1} + sqrt(1 - rho^2) * innovation_t

    Args:
        innovations: shape (..., T_max) — the raw daily draws.
        rho: AR(1) autocorrelation parameter.

    Returns:
        Correlated series, same shape as innovations.
    """
    if rho == 0.0:
        return innovations

    T_max = innovations.shape[-1]
    out = np.empty_like(innovations)
    scale = np.sqrt(1.0 - rho * rho)

    # First element: scaled innovation (no prior)
    out[..., 0] = scale * innovations[..., 0]

    # Recursive AR(1)
    for t in range(1, T_max):
        out[..., t] = rho * out[..., t - 1] + scale * innovations[..., t]

    return out
