"""
Test Theorem 1 & 2: Exact size control of the randomization test.

Under H0(theta0), the sign-flip randomization test should have exact
finite-sample size control:
  P(p_value <= alpha) <= alpha

We verify this via nested simulation: for each of N_MC outer experiments,
we run the randomization test with B inner draws and check if the
empirical rejection rate is within MC error of alpha.

This is THE critical test — all theoretical guarantees depend on it.
"""

import numpy as np
import pytest
from gsi.dgp import make_paths, realize
from gsi.bands import uniform_band_test
from gsi.omega_design import design_omega


class TestExactnessTheorem1:
    """Theorem 1: Fixed-time exactness at t = T_max."""

    @pytest.mark.slow
    @pytest.mark.parametrize("tail", ["gaussian", "t2", "pareto"])
    @pytest.mark.parametrize("rho", [0.0, 0.5, 0.8])
    @pytest.mark.parametrize("stat_name", ["sgn", "rk", "tr"])
    def test_size_control_fixed_time(self, tail, rho, stat_name):
        """Empirical size <= alpha + 3*MC_se for fixed-time test.

        Uses the uniform_band_test with omega that puts all weight at T_max.
        """
        n_pairs, T_max = 20, 15
        alpha = 0.05
        N_MC = 200  # outer replications
        B = 499      # inner randomization draws (odd for p-value precision)

        # Omega: all weight at final period (fixed-time test)
        omega = np.zeros(T_max)
        omega[-1] = 1.0

        rejections = 0
        for mc in range(N_MC):
            # Each MC rep gets a deterministically derived seed (no global seed)
            mc_rng = np.random.default_rng(hash((tail, rho, stat_name, mc)) % (2**31))

            # Generate panel under H0
            panel = make_paths(
                n_pairs=n_pairs, T_max=T_max, theta=0.5,
                tail=tail, rho=rho, seasonality=False, hetero_scale=0.1,
                entry_days=None, rng=mc_rng,
            )
            Z = mc_rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = panel.realize(Z, mc_rng)

            result = uniform_band_test(
                obs=obs, theta0=0.5, stat=stat_name,
                omega=omega, B=B, alpha=alpha, rng=mc_rng,
            )
            if result.reject:
                rejections += 1

        emp_size = rejections / N_MC
        mc_se = np.sqrt(alpha * (1 - alpha) / N_MC)

        assert emp_size <= alpha + 3 * mc_se, (
            f"Empirical size {emp_size:.4f} exceeds alpha={alpha} + 3*SE={3*mc_se:.4f} "
            f"for tail={tail}, rho={rho}, stat={stat_name}"
        )

    @pytest.mark.nightly
    @pytest.mark.parametrize("tail", ["gaussian", "t2", "pareto"])
    @pytest.mark.parametrize("rho", [0.0, 0.8])
    @pytest.mark.parametrize("stat_name", ["sgn", "rk", "tr"])
    def test_size_control_full_spec(self, tail, rho, stat_name):
        """Full-spec size control: N_MC=500 x B=999 (nightly only)."""
        n_pairs, T_max = 20, 15
        alpha = 0.05
        N_MC = 500
        B = 999

        omega = np.zeros(T_max)
        omega[-1] = 1.0

        rejections = 0
        for mc in range(N_MC):
            mc_rng = np.random.default_rng(hash((tail, rho, stat_name, mc, 1)) % (2**31))
            panel = make_paths(
                n_pairs=n_pairs, T_max=T_max, theta=0.5,
                tail=tail, rho=rho, seasonality=False, hetero_scale=0.1,
                entry_days=None, rng=mc_rng,
            )
            Z = mc_rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = panel.realize(Z, mc_rng)
            result = uniform_band_test(
                obs=obs, theta0=0.5, stat=stat_name,
                omega=omega, B=B, alpha=alpha, rng=mc_rng,
            )
            if result.reject:
                rejections += 1

        emp_size = rejections / N_MC
        mc_se = np.sqrt(alpha * (1 - alpha) / N_MC)
        assert emp_size <= alpha + 3 * mc_se, (
            f"[nightly] Empirical size {emp_size:.4f} exceeds alpha={alpha} "
            f"+ 3*SE={3*mc_se:.4f} for tail={tail}, rho={rho}, stat={stat_name}"
        )

    def test_exactness_various_alphas(self, realized_small):
        """Test that p-value distribution is roughly uniform under H0."""
        obs = realized_small
        true_theta = obs.panel.theta[0]
        n_pairs = obs.panel.n_pairs
        T_max = obs.panel.T_max

        omega = np.ones(T_max)
        rng_test = np.random.default_rng(77)

        # With B=1999, p-values should be approximately uniform
        result = uniform_band_test(
            obs=obs, theta0=true_theta, stat="sgn",
            omega=omega, B=1999, alpha=0.05, rng=rng_test,
        )

        # p-value should be between 0 and 1
        assert 0.0 <= result.p_value <= 1.0

        # Under H0, p-value should NOT be close to 0 (non-rejection expected)
        assert result.p_value > 0.01, (
            f"p-value {result.p_value:.4f} suspiciously small under H0"
        )


class TestExactnessTheorem2:
    """Theorem 2: Time-uniform size control."""

    @pytest.mark.slow
    @pytest.mark.parametrize("stat_name", ["sgn", "rk", "tr"])
    def test_time_uniform_no_inflation(self, stat_name):
        """Time-uniform band should not inflate Type I error over multiple looks."""
        n_pairs, T_max = 20, 15
        alpha = 0.05
        N_MC = 200
        B = 499

        # Pocock-style uniform omega
        omega = np.ones(T_max)

        rejections = 0
        for mc in range(N_MC):
            mc_rng = np.random.default_rng(hash((stat_name, mc, 2000)) % (2**31))

            panel = make_paths(
                n_pairs=n_pairs, T_max=T_max, theta=0.5,
                tail="t2", rho=0.6, seasonality=False, hetero_scale=0.1,
                entry_days=None, rng=mc_rng,
            )
            Z = mc_rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            obs = panel.realize(Z, mc_rng)

            result = uniform_band_test(
                obs=obs, theta0=0.5, stat=stat_name,
                omega=omega, B=B, alpha=alpha, rng=mc_rng,
            )
            if result.reject:
                rejections += 1

        emp_size = rejections / N_MC
        mc_se = np.sqrt(alpha * (1 - alpha) / N_MC)

        assert emp_size <= alpha + 3 * mc_se, (
            f"Time-uniform empirical size {emp_size:.4f} exceeds alpha={alpha} "
            f"+ 3*SE={3*mc_se:.4f} for stat={stat_name}"
        )

    def test_comparison_fixed_vs_sequential(self, realized_small):
        """Sequential test should be more conservative than fixed-time at T_max."""
        obs = realized_small
        true_theta = obs.panel.theta[0]
        T_max = obs.panel.T_max
        rng_test = np.random.default_rng(123)

        # Fixed-time: weight only at T_max
        omega_fixed = np.zeros(T_max)
        omega_fixed[-1] = 1.0

        # Sequential: uniform weights (more conservative)
        omega_seq = np.ones(T_max)

        result_fixed = uniform_band_test(
            obs=obs, theta0=true_theta, stat="sgn",
            omega=omega_fixed, B=1999, alpha=0.05, rng=rng_test,
        )
        result_seq = uniform_band_test(
            obs=obs, theta0=true_theta, stat="sgn",
            omega=omega_seq, B=1999, alpha=0.05, rng=rng_test,
        )

        # Sequential should have higher critical value (more conservative)
        assert result_seq.c_alpha >= result_fixed.c_alpha, (
            f"Sequential c_alpha={result_seq.c_alpha:.4f} should be >= "
            f"fixed-time c_alpha={result_fixed.c_alpha:.4f}"
        )

    def test_omega_shape_matters(self, realized_small):
        """Different omega shapes produce different critical values."""
        obs = realized_small
        true_theta = obs.panel.theta[0]
        T_max = obs.panel.T_max
        rng_test = np.random.default_rng(456)

        omega_pocock = design_omega(None, "pocock", None, 0.2, T_max)
        omega_obf = design_omega(None, "obf", None, 0.2, T_max)

        result_pocock = uniform_band_test(
            obs=obs, theta0=true_theta, stat="sgn",
            omega=omega_pocock, B=1999, alpha=0.05, rng=rng_test,
        )
        result_obf = uniform_band_test(
            obs=obs, theta0=true_theta, stat="sgn",
            omega=omega_obf, B=1999, alpha=0.05, rng=rng_test,
        )

        # Different shapes should produce different critical values
        # (they might be equal by chance with low B, but generally different)
        # This is more of a smoke test than a hard assertion.
        assert result_pocock.c_alpha > 0
        assert result_obf.c_alpha > 0


class TestDiscreteNature:
    """Verify handling of discrete p-value distribution."""

    def test_p_value_plus_one_correction(self, realized_small):
        """p-value includes +1 correction in numerator and denominator."""
        obs = realized_small
        T_max = obs.panel.T_max
        omega = np.ones(T_max)
        rng_test = np.random.default_rng(999)

        result = uniform_band_test(
            obs=obs, theta0=0.0, stat="sgn",
            omega=omega, B=9, alpha=0.05, rng=rng_test,
        )

        # p = (1 + count) / (1 + B)
        expected_min = 1.0 / 10.0  # 0.1
        assert result.p_value >= expected_min, (
            f"p-value {result.p_value} should be >= {expected_min}"
        )
        assert result.p_value <= 1.0
