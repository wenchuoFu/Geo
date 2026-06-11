"""
Residual computation for paired geo experiments.

Under the sharp null H0(theta0), the residual matrix has the sign-symmetry
property (Lemma 1): eps(theta0) = Z[:,None] * e where e is a fixed matrix
that does not depend on the treatment assignment Z.
"""

import numpy as np
from . import ObsPanel


def residual_matrix(obs: ObsPanel, theta0: float) -> np.ndarray:
    """Compute residual matrix under hypothesized iROAS theta0.

    epsilon_i(theta0, t) = dY_i(t) - theta0 * dS_i(t)

    Args:
        obs: Realized panel with observed differences dY, dS.
        theta0: Hypothesized constant iROAS value.

    Returns:
        Residual matrix, shape (n_pairs, T_max). dtype float64.

    Notes:
        Under H0(theta0) where theta0 equals the true theta, Lemma 1 gives:
          epsilon_i(theta0, t) = Z_i * e_i(t)
        where e_i(t) is a fixed path not depending on Z.
        This is the foundation for all sign-flip randomization tests.
    """
    eps = obs.dY.astype(np.float64) - theta0 * obs.dS.astype(np.float64)
    return eps
