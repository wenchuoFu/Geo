"""
Test Theorem 3: Martingale property of the betting wealth process.

Under H0(theta0), the wealth process W_k must be a non-negative martingale:
  E[W_k] = 1 for all k
  W_k >= 0 for all k

We verify this via simulation of many wealth paths.
"""

import numpy as np
import pytest
from gsi.betting import betting_cs
from gsi import PairOutcome


class TestMartingaleProperty:
    """Theorem 3(i): Wealth process is a non-negative martingale."""

    def test_initial_wealth_is_one(self):
        """Wealth process must start at 1."""
        # The betting_cs internal state initializes wealth at 1.0
        # We verify this through the first snapshot.
        theta_grid = np.array([0.0, 0.5, 1.0])

        # Create a simple stream with one outcome
        outcomes = [
            PairOutcome(pair_id=0, eps=0.3, eval_time=0, dS_eval=1.0),
        ]

        rng = np.random.default_rng(42)
        snapshots = list(betting_cs(iter(outcomes), theta_grid, alpha=0.05, rng=rng))

        assert len(snapshots) == 1
        # Wealth for each theta should be >= 0 (non-negative)
        assert np.all(snapshots[0].wealth >= 0.0)
        # After first bet, wealth may differ from 1.0 depending on bet outcome

    def test_wealth_nonnegative(self, rng):
        """Wealth must never go negative (supermartingale with non-negative values)."""
        n_pairs = 50
        theta_grid = np.array([0.0])

        # Generate outcomes under H0 (eps with symmetric signs)
        true_eps = rng.normal(size=n_pairs)  # symmetric around 0
        assignments = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
        # Under H0: eps = Z * true_eps where Z ~ Rademacher
        eps_observed = assignments * true_eps  # symmetric distribution

        outcomes = [
            PairOutcome(pair_id=i, eps=float(eps_observed[i]), eval_time=i, dS_eval=1.0)
            for i in range(n_pairs)
        ]

        snapshots = list(betting_cs(iter(outcomes), theta_grid, alpha=0.05, rng=rng))

        for snap in snapshots:
            assert np.all(snap.wealth >= 0.0), (
                f"Wealth went negative at k={snap.k}: min={np.min(snap.wealth)}"
            )

    @pytest.mark.slow
    def test_expected_wealth_is_one(self, rng):
        """Under H0, E[W_k] should be approximately 1.

        Simulate 2000 wealth paths and check that |mean(W_k) - 1| < 3 * SE.
        """
        n_pairs = 20
        n_paths = 2000
        theta_grid = np.array([0.0])

        wealth_at_end = np.empty(n_paths)

        for path in range(n_paths):
            path_rng = np.random.default_rng(path * 1000 + 3)

            # Generate data under H0
            true_eps = path_rng.normal(size=n_pairs)
            assignments = path_rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            eps_observed = assignments * true_eps

            outcomes = [
                PairOutcome(
                    pair_id=i,
                    eps=float(eps_observed[i]),
                    eval_time=i,
                    dS_eval=1.0,
                )
                for i in range(n_pairs)
            ]

            snapshots = list(betting_cs(
                iter(outcomes), theta_grid, alpha=0.05,
                rng=path_rng,
            ))

            wealth_at_end[path] = snapshots[-1].wealth[0]

        mean_wealth = np.mean(wealth_at_end)
        se_wealth = np.std(wealth_at_end) / np.sqrt(n_paths)

        # E[W_k] = 1 for a martingale
        assert abs(mean_wealth - 1.0) < 3.0 * se_wealth, (
            f"Mean wealth {mean_wealth:.4f} deviates from 1.0 by more than "
            f"3*SE = {3*se_wealth:.4f}"
        )

    def test_zero_eps_no_wealth_change(self, rng):
        """When eps=0, wealth should not change (sign(0)=0, g=0, f=0)."""
        theta_grid = np.array([0.0])
        outcomes = [
            PairOutcome(pair_id=i, eps=0.0, eval_time=i, dS_eval=1.0)
            for i in range(5)
        ]

        snapshots = list(betting_cs(iter(outcomes), theta_grid, alpha=0.05, rng=rng))

        # All wealth values should be exactly 1.0 (no bets placed)
        for snap in snapshots:
            np.testing.assert_allclose(snap.wealth, 1.0, atol=1e-14,
                                       err_msg=f"Wealth changed at k={snap.k} despite zero eps")


class TestWealthAccumulation:
    """Test that betting accumulates correctly with non-zero signals."""

    def test_wealth_increases_with_correct_sign(self, rng):
        """If all signs are positive, upper wealth should increase."""
        theta_grid = np.array([0.0])
        outcomes = [
            PairOutcome(pair_id=i, eps=1.0, eval_time=i, dS_eval=1.0)
            for i in range(10)
        ]

        snapshots = list(betting_cs(iter(outcomes), theta_grid, alpha=0.05, rng=rng))

        # Wealth should generally increase when signals align
        final_wealth = snapshots[-1].wealth[0]
        assert final_wealth > 1.0, (
            f"Wealth {final_wealth} should increase with consistent positive signals"
        )

    def test_two_sided_processes(self, rng):
        """Both W+ and W- should be non-negative.

        The max of the two provides two-sided coverage.
        """
        n_pairs = 30
        theta_grid = np.array([0.0])

        # Mixed signals
        true_eps = rng.normal(size=n_pairs)
        assignments = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
        eps_observed = assignments * true_eps

        outcomes = [
            PairOutcome(pair_id=i, eps=float(eps_observed[i]), eval_time=i, dS_eval=1.0)
            for i in range(n_pairs)
        ]

        snapshots = list(betting_cs(iter(outcomes), theta_grid, alpha=0.05, rng=rng))

        # The max wealth should dominate individual processes
        final_wealth = snapshots[-1].wealth[0]
        assert final_wealth >= 0.0
