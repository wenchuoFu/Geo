"""
Baseline methods for comparison in experiments.

These are reference implementations of standard approaches, used to
demonstrate the advantages of the exact randomization-based methods
(Theorem 2 and Theorem 3) over:

  - tm_wrapper: Google's trimmed_match (fixed-time CI)
  - catoni_cs: Catoni-type confidence sequence (assumes iid increments)
  - hoeffding_cs: Sub-Gaussian mixture CS (fails under heavy tails)
  - naive_peeking: Daily t-test with early stopping (inflates Type I error)

All baseline wrappers are optional — tm_wrapper requires the trimmed_match
package; the others are pure numpy/scipy.
"""

from .tm_wrapper import tm_ci, is_tm_available
from .catoni_cs import catoni_cs
from .hoeffding_cs import hoeffding_cs
from .naive_peeking import naive_peeking_test

__all__ = [
    "tm_ci", "is_tm_available",
    "catoni_cs",
    "hoeffding_cs",
    "naive_peeking_test",
]
