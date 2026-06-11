"""
Shared fixtures for gsi tests.
"""

import pytest
import numpy as np


@pytest.fixture
def rng():
    """Deterministic random generator (seed=42)."""
    return np.random.default_rng(42)


@pytest.fixture
def rng_fixed():
    """Alternative seed for independent draws."""
    return np.random.default_rng(12345)


@pytest.fixture
def small_panel_h0(rng):
    """Small GeoPanel under sharp null H0(theta=0.5).

    n=10 pairs, T=20 days, Gaussian tails, moderate AR(1).
    All pairs enter at t=0.
    """
    from gsi.dgp import make_paths
    return make_paths(
        n_pairs=10,
        T_max=20,
        theta=0.5,
        tail="gaussian",
        rho=0.3,
        seasonality=False,
        hetero_scale=0.0,
        entry_days=None,
        rng=rng,
    )


@pytest.fixture
def medium_panel_h0(rng):
    """Medium GeoPanel, n=30, T=30, t2 tails, high AR(1)."""
    from gsi.dgp import make_paths
    return make_paths(
        n_pairs=30,
        T_max=30,
        theta=0.5,
        tail="t2",
        rho=0.7,
        seasonality=True,
        hetero_scale=0.1,
        entry_days=None,
        rng=rng,
    )


@pytest.fixture
def realized_small(small_panel_h0, rng_fixed):
    """Realize the small panel with a random Z assignment."""
    from gsi.dgp import realize
    Z = rng_fixed.choice(np.array([-1, 1], dtype=np.int8), size=small_panel_h0.n_pairs)
    return realize(small_panel_h0, Z)


@pytest.fixture
def realized_medium(medium_panel_h0, rng_fixed):
    """Realize the medium panel with a random Z assignment."""
    from gsi.dgp import realize
    Z = rng_fixed.choice(np.array([-1, 1], dtype=np.int8), size=medium_panel_h0.n_pairs)
    return realize(medium_panel_h0, Z)
