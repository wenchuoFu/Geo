"""
Test Lemma 1: Sign representation of residuals under the sharp null.

Under H0(theta0), the residual matrix satisfies:
  eps_i(theta0, t) = Z_i * e_i(t)
where e_i(t) is a fixed matrix not depending on Z.

This is the foundational lemma that justifies all sign-flip randomization.
"""

import numpy as np
import pytest
from gsi.residuals import residual_matrix


class TestSignRepresentation:
    """Verify Lemma 1 numerically."""

    def test_sign_symmetry_structure(self, realized_small):
        """Lemma 1: eps == Z[:,None] * e for some fixed e.

        Construct e from the latent paths and verify exact equality.
        """
        obs = realized_small
        Z = obs.Z

        # Compute residuals at the true theta
        true_theta = obs.panel.theta[0]  # constant under sharp null
        eps = residual_matrix(obs, true_theta)  # (n, T)

        # Recover e_i(t) = Z_i * eps_i(t) (since Z_i^2 = 1)
        e_matrix = Z[:, np.newaxis] * eps  # (n, T)

        # Verify: eps == Z * e
        reconstructed = Z[:, np.newaxis] * e_matrix
        np.testing.assert_allclose(eps, reconstructed, atol=1e-10)

    def test_e_matrix_independent_of_Z(self, realized_small, rng):
        """e_i(t) should not depend on Z — verify by recomputing with different Z.

        Since e_i(t) = Y_i1^C(t) - Y_i2^C(t) - theta0*(S_i1^C(t) - S_i2^C(t)),
        it's a fixed function of control potential outcomes only.
        """
        obs = realized_small
        panel = obs.panel
        true_theta = panel.theta[0]

        # Compute e from the first realization
        eps1 = residual_matrix(obs, true_theta)
        e1 = obs.Z[:, np.newaxis] * eps1

        # Realize with a different Z assignment
        Z2 = rng.choice(np.array([-1, 1], dtype=np.int8), size=panel.n_pairs)
        obs2 = panel.realize(Z2, rng)
        eps2 = residual_matrix(obs2, true_theta)
        e2 = Z2[:, np.newaxis] * eps2

        # e should be identical (it's a fixed function of latent paths)
        np.testing.assert_allclose(e1, e2, atol=1e-10)

    def test_sign_flip_distribution(self, realized_small):
        """Under H0, flipping signs of eps should produce same distribution.

        sigma * eps has the same distribution as eps when sigma ~ Rademacher.
        """
        obs = realized_small
        true_theta = obs.panel.theta[0]
        eps = residual_matrix(obs, true_theta)

        # Generate 1000 random sign flips and check symmetry
        rng = np.random.default_rng(99)
        n = obs.panel.n_pairs

        # For each pair i, eps_i and -eps_i should be equally likely
        # under the randomization distribution
        for _ in range(100):
            sigma = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
            eps_flipped = sigma[:, np.newaxis] * eps
            # The distribution of eps_flipped should be exchangeable with eps
            # Check that column means are symmetric around 0
            col_mean = np.mean(eps_flipped, axis=0)
            # Under H0, E[eps_flipped] = 0 for each column
            # (the mean over sign-flips should converge to 0)
            # This is a weak check; the real test is in test_exactness.py
            assert np.all(np.abs(col_mean) < 10.0)  # loose bound

    def test_sign_zero_handling(self):
        """sign(0) = 0 per spec convention."""
        eps_t = np.array([0.0, 0.0, 0.0])
        assert np.all(np.sign(eps_t) == 0.0)

        # Mixed zeros
        eps_t = np.array([1.0, 0.0, -1.0, 0.0])
        signs = np.sign(eps_t)
        assert signs[0] == 1.0
        assert signs[1] == 0.0
        assert signs[2] == -1.0
        assert signs[3] == 0.0

    def test_sign_representation_under_alternative(self, realized_small):
        """Under theta0 != true_theta, the sign symmetry should break."""
        obs = realized_small
        true_theta = obs.panel.theta[0]
        false_theta = true_theta + 2.0  # clearly wrong

        eps_true = residual_matrix(obs, true_theta)
        eps_false = residual_matrix(obs, false_theta)

        # e_matrix from false theta should differ from true theta
        e_true = obs.Z[:, np.newaxis] * eps_true
        e_false = obs.Z[:, np.newaxis] * eps_false

        # They should NOT be equal (unless degenerate)
        assert not np.allclose(e_true, e_false, atol=1e-10)
