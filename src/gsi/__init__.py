"""
gsi — Geo Sequential Inference

Exact randomization-based inference for iROAS in paired geo experiments.
Implements Theorem 2 (time-uniform confidence bands) and Theorem 3 (staggered-
entry betting confidence sequences) from the paired geo inference specification.

All randomness flows through explicit `rng: np.random.Generator` parameters.
No global seeds are used anywhere in the library.
"""

# ── Dataclasses ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Optional, Iterator
import numpy as np


@dataclass
class GeoPanel:
    """Pre-randomization latent paths (design-based, all fixed constants).

    Each pair i has two geos (index 0, 1). Z_i = +1 means geo 0 is treated.
    All paths are cumulative over time t = 0, ..., T_max-1.

    Attributes:
        n_pairs: Number of geo pairs.
        T_max: Number of time periods (days).
        Y_C: Control response paths, shape (n_pairs, 2, T_max).
        S_C: Control spend paths, shape (n_pairs, 2, T_max).
        delta_S: Spend increment under treatment (S^T - S^C), shape (n_pairs, 2, T_max).
        theta: True per-pair iROAS, shape (n_pairs,). Constant under sharp null.
        entry_days: Day each pair enters the experiment (0-indexed), shape (n_pairs,).
            Periods before entry_day[i] are pre-period / not-yet-entered.
    """
    n_pairs: int
    T_max: int
    Y_C: np.ndarray          # (n_pairs, 2, T_max)
    S_C: np.ndarray          # (n_pairs, 2, T_max)
    delta_S: np.ndarray      # (n_pairs, 2, T_max)
    theta: np.ndarray        # (n_pairs,)
    entry_days: np.ndarray   # (n_pairs,)

    def __post_init__(self):
        for arr_name in ("Y_C", "S_C", "delta_S"):
            arr = getattr(self, arr_name)
            if arr.shape != (self.n_pairs, 2, self.T_max):
                raise ValueError(
                    f"{arr_name} expected shape ({self.n_pairs}, 2, {self.T_max}), "
                    f"got {arr.shape}"
                )
        if self.theta.shape != (self.n_pairs,):
            raise ValueError(
                f"theta expected shape ({self.n_pairs},), got {self.theta.shape}"
            )
        if self.entry_days.shape != (self.n_pairs,):
            raise ValueError(
                f"entry_days expected shape ({self.n_pairs},), "
                f"got {self.entry_days.shape}"
            )

    def realize(self, Z: np.ndarray, rng: np.random.Generator) -> "ObsPanel":
        """Map treatment assignment Z to observed panel.

        Args:
            Z: Assignment vector, shape (n_pairs,). +1 means geo 0 treated.
            rng: Random generator (unused for the deterministic mapping itself,
                 but included for interface consistency).

        Returns:
            ObsPanel with dY and dS differences.
        """
        from .dgp import realize
        return realize(self, Z)


@dataclass
class ObsPanel:
    """Realized panel after treatment assignment Z is applied.

    Attributes:
        panel: The underlying GeoPanel (latent paths, theta, entry_days).
        Z: Treatment assignment, shape (n_pairs,). +1 or -1.
        dY: Observed cumulative response difference treated - control,
            shape (n_pairs, T_max).
        dS: Observed cumulative spend difference treated - control,
            shape (n_pairs, T_max).
    """
    panel: GeoPanel
    Z: np.ndarray            # (n_pairs,)
    dY: np.ndarray           # (n_pairs, T_max)
    dS: np.ndarray           # (n_pairs, T_max)

    def __post_init__(self):
        n, T = self.panel.n_pairs, self.panel.T_max
        for arr_name in ("Z",):
            arr = getattr(self, arr_name)
            if arr.shape != (n,):
                raise ValueError(
                    f"{arr_name} expected shape ({n},), got {arr.shape}"
                )
        for arr_name in ("dY", "dS"):
            arr = getattr(self, arr_name)
            if arr.shape != (n, T):
                raise ValueError(
                    f"{arr_name} expected shape ({n}, {T}), got {arr.shape}"
                )


@dataclass
class BandResult:
    """Output of uniform_band_test for a single theta0.

    Attributes:
        reject: Whether H0(theta0) is rejected at level alpha.
        p_value: MC p-value with +1 correction.
        c_alpha: Critical value (1-alpha quantile of null distribution).
        M_obs: Observed max-of-weighted-statistics.
        M_null: Null distribution draws, shape (B,).
        theta0: The hypothesized iROAS value tested.
        alpha: Significance level.
    """
    reject: bool
    p_value: float
    c_alpha: float
    M_obs: float
    M_null: np.ndarray       # (B,)
    theta0: float
    alpha: float


@dataclass
class CSResult:
    """Output of confidence_set inversion over a theta grid.

    Attributes:
        lower: Lower bound of confidence set (None if unbounded below).
        upper: Upper bound of confidence set (None if unbounded above).
        accepted: Boolean mask over theta_grid for accepted values.
        theta_grid: The grid of theta values searched.
        p_values: p-value at each grid point.
        sequential: Whether sequential/time-uniform inference was used.
        confidence_level: 1 - alpha.
        is_interval: True if the acceptance set is a single contiguous interval.
        unbounded: True if Fieller set is unbounded (endpoints accepted).
    """
    lower: Optional[float]
    upper: Optional[float]
    accepted: np.ndarray       # (len(theta_grid),) bool
    theta_grid: np.ndarray     # (len(theta_grid),)
    p_values: np.ndarray       # (len(theta_grid),)
    sequential: bool
    confidence_level: float
    is_interval: bool
    unbounded: bool


@dataclass
class PairOutcome:
    """Per-pair outcome for staggered-entry betting (Theorem 3).

    Attributes:
        pair_id: Index of the pair (0-based).
        eps: The residual ε_k(θ₀) = dY_k - θ₀ * dS_k at the pair's evaluation window
            (scalar — cumulative over the pair's full evaluation period).
        eval_time: The time index (end of eval window) when this pair is settled.
        dS_eval: Cumulative spend difference used for this pair's eps.
    """
    pair_id: int
    eps: float
    eval_time: int
    dS_eval: float


@dataclass
class CSSnapshot:
    """A single snapshot of the betting confidence sequence at evaluation step k.

    Attributes:
        k: Number of pairs settled so far (1-indexed).
        eval_time: Time index of the most recently settled pair.
        lower: Lower bound of current CS (None if unbounded below).
        upper: Upper bound of current CS (None if unbounded above).
        wealth: Current max wealth per theta0, shape (len(theta_grid),).
        accepted: Boolean mask of non-rejected theta0 values.
    """
    k: int
    eval_time: int
    lower: Optional[float]
    upper: Optional[float]
    wealth: np.ndarray        # (len(theta_grid),)
    accepted: np.ndarray      # (len(theta_grid),) bool


# ── Public API ────────────────────────────────────────────────────────────────

from .dgp import make_paths
from .residuals import residual_matrix
from .statistics import t_sgn, t_rk, t_tr
from .randomization import flip_distribution
from .bands import uniform_band_test
from .inversion import confidence_set
from .betting import betting_cs
from .spec_test import spec_test
from .omega_design import design_omega

__all__ = [
    # Dataclasses
    "GeoPanel", "ObsPanel", "BandResult", "CSResult", "PairOutcome", "CSSnapshot",
    # Core functions
    "make_paths",
    "residual_matrix",
    "t_sgn", "t_rk", "t_tr",
    "flip_distribution",
    "uniform_band_test",
    "confidence_set",
    "betting_cs",
    "spec_test",
    "design_omega",
]
