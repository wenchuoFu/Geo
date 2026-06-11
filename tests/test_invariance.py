"""
Test flip-invariance of precomputed quantities.

Ranks of |eps| and MAD(|eps|) must be invariant under sign flips.
This is critical for caching correctness in the randomization engine.
"""

import numpy as np
import pytest
from gsi.statistics import (
    precompute_ranks,
    precompute_mad,
    precompute_sign_contrib,
    precompute_rk_contrib,
)


class TestFlipInvariance:
    """Verify that rank and MAD precomputation is flip-invariant."""

    def test_rank_invariance(self, rng):
        """ranks(|eps|) == ranks(|sigma * eps|) for any sigma in {-1,+1}^n."""
        n, T = 20, 10
        eps = rng.normal(size=(n, T))
        sigma = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)

        ranks_orig = precompute_ranks(eps)
        eps_flipped = sigma[:, np.newaxis] * eps
        ranks_flipped = precompute_ranks(eps_flipped)

        np.testing.assert_allclose(ranks_orig, ranks_flipped, atol=1e-14)

    def test_mad_invariance(self, rng):
        """MAD(|eps|) == MAD(|sigma * eps|) for any sigma in {-1,+1}^n."""
        n, T = 20, 10
        eps = rng.normal(size=(n, T))
        sigma = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)

        mad_orig = precompute_mad(eps)
        eps_flipped = sigma[:, np.newaxis] * eps
        mad_flipped = precompute_mad(eps_flipped)

        np.testing.assert_allclose(mad_orig, mad_flipped, atol=1e-14)

    def test_sign_contrib_flips_correctly(self, rng):
        """sign(sigma * eps) == sigma * sign(eps) elementwise."""
        n, T = 20, 10
        eps = rng.normal(size=(n, T))
        sigma = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)

        contrib_orig = precompute_sign_contrib(eps)
        eps_flipped = sigma[:, np.newaxis] * eps
        contrib_flipped = precompute_sign_contrib(eps_flipped)

        expected = sigma[:, np.newaxis] * contrib_orig
        np.testing.assert_allclose(contrib_flipped, expected, atol=1e-14)

    def test_rk_contrib_flips_correctly(self, rng):
        """rank_sign(sigma * eps) == sigma * rank_sign(eps) since ranks invariant."""
        n, T = 20, 10
        eps = rng.normal(size=(n, T))
        sigma = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)

        contrib_orig = precompute_rk_contrib(eps)
        eps_flipped = sigma[:, np.newaxis] * eps
        contrib_flipped = precompute_rk_contrib(eps_flipped)

        expected = sigma[:, np.newaxis] * contrib_orig
        np.testing.assert_allclose(contrib_flipped, expected, atol=1e-14)

    def test_multiple_flips_consistent(self, rng):
        """Multiple random flips should all produce same ranks and MAD."""
        n, T = 20, 10
        eps = rng.normal(size=(n, T))

        ranks_ref = precompute_ranks(eps)
        mad_ref = precompute_mad(eps)

        for _ in range(50):
            sigma = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
            eps_f = sigma[:, np.newaxis] * eps
            np.testing.assert_allclose(precompute_ranks(eps_f), ranks_ref, atol=1e-14)
            np.testing.assert_allclose(precompute_mad(eps_f), mad_ref, atol=1e-14)

    def test_zero_residuals_mad(self):
        """MAD of all-zero residuals should produce small positive sentinel."""
        eps = np.zeros((10, 5))
        mad = precompute_mad(eps)
        # Should be >= 1e-12 (sentinel)
        assert np.all(mad >= 1e-12)

    def test_constant_residuals_ranks(self):
        """All equal |eps| should produce uniform ranks."""
        eps = np.full((10, 5), 5.0)
        ranks = precompute_ranks(eps)
        # All ranks should be equal; sum of ranks per column = n*(n+1)/2n
        expected_rank = np.mean(np.arange(1, 11)) / 10  # 0.55 for n=10
        np.testing.assert_allclose(ranks, expected_rank, atol=1e-10)
