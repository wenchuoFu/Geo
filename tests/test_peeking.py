"""
Test any-time stopping coverage (Theorem 2 & 3).

The key guarantee of time-uniform methods is that coverage holds at ANY
stopping time — even if the stopping rule depends on the observed data.

We verify this by simulating with a "stop when the statistic looks promising"
rule and checking that the true theta is still covered with probability >= 1-alpha.
"""

import numpy as np
import pytest
from gsi.betting import betting_cs
from gsi import PairOutcome


class TestAnytimeStopping:
    """Any-time stopping does not inflate Type I error."""

    def test_fixed_stopping_time_coverage(self, rng):
        """At a fixed k, coverage should be >= 1-alpha."""
        n_pairs = 30
        alpha = 0.05
        N_MC = 500
        theta_grid = np.linspace(-1.0, 1.0, 51)

        coverage_count = 0
        true_theta = 0.0

        for mc in range(N_MC):
            mc_rng = np.random.default_rng(mc * 100 + 5)

            # Generate data under H0 at theta=0
            true_eps = mc_rng.normal(size=n_pairs)
            assignments = mc_rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            eps_observed = assignments * true_eps

            outcomes = [
                PairOutcome(
                    pair_id=i, eps=float(eps_observed[i]), eval_time=i, dS_eval=1.0
                )
                for i in range(n_pairs)
            ]

            snapshots = list(betting_cs(
                iter(outcomes), theta_grid, alpha, rng=mc_rng,
            ))

            # Check coverage at final snapshot
            final = snapshots[-1]
            if (final.lower is None or final.lower <= true_theta) and \
               (final.upper is None or final.upper >= true_theta):
                coverage_count += 1

        emp_coverage = coverage_count / N_MC
        mc_se = np.sqrt(alpha * (1 - alpha) / N_MC)

        assert emp_coverage >= (1 - alpha) - 3 * mc_se, (
            f"Coverage {emp_coverage:.4f} below nominal {1-alpha} - 3*SE={3*mc_se:.4f}"
        )

    @pytest.mark.slow
    def test_data_dependent_stopping_coverage(self, rng):
        """Coverage holds when stopping rule depends on data.

        Stopping rule: stop when wealth > 2 (early significance signal).
        Coverage should still be >= 1-alpha at stopping time.
        """
        n_pairs = 40
        alpha = 0.05
        N_MC = 500
        theta_grid = np.linspace(-1.0, 1.0, 51)
        true_theta = 0.0

        coverage_count = 0

        for mc in range(N_MC):
            mc_rng = np.random.default_rng(mc * 200 + 11)

            true_eps = mc_rng.normal(size=n_pairs)
            assignments = mc_rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
            eps_observed = assignments * true_eps

            outcomes = [
                PairOutcome(
                    pair_id=i, eps=float(eps_observed[i]), eval_time=i, dS_eval=1.0
                )
                for i in range(n_pairs)
            ]

            # Simulate with data-dependent stopping
            cs_iter = betting_cs(iter(outcomes), theta_grid, alpha, rng=mc_rng)

            stopped_snapshot = None
            for snapshot in cs_iter:
                if np.any(snapshot.wealth > 2.0):
                    stopped_snapshot = snapshot
                    break

            if stopped_snapshot is None:
                # Never stopped — use last (we consumed the iterator, so reconstruct)
                # Just use final: coverage at never-stopped is conservative
                cs_iter2 = betting_cs(
                    iter(outcomes), theta_grid, alpha, rng=mc_rng
                )
                for snapshot in cs_iter2:
                    stopped_snapshot = snapshot
                # last one

            if stopped_snapshot is not None:
                if (stopped_snapshot.lower is None or stopped_snapshot.lower <= true_theta) and \
                   (stopped_snapshot.upper is None or stopped_snapshot.upper >= true_theta):
                    coverage_count += 1

        emp_coverage = coverage_count / N_MC
        mc_se = np.sqrt(alpha * (1 - alpha) / N_MC)

        assert emp_coverage >= (1 - alpha) - 3 * mc_se, (
            f"Coverage under data-dependent stopping: {emp_coverage:.4f} < "
            f"nominal {1-alpha} - 3*SE={3*mc_se:.4f}"
        )

    def test_vs_naive_peeking_comparison(self, realized_small):
        """Naive peeking should show inflated rejection, but Theorem 2 should not.

        This is a demonstration test: the naive t-test on ratios should have
        inflated Type I error, while our sequential method controls it.
        """
        from gsi.baselines.naive_peeking import naive_peeking_test

        obs = realized_small
        naive_result = naive_peeking_test(obs.dY, obs.dS, alpha=0.05)

        # With small sample, naive may or may not reject (this is just documentation)
        # The real comparison is in Experiment E2 with many replications.
        assert "reject" in naive_result
        assert "first_crossing" in naive_result
        assert "p_values" in naive_result
