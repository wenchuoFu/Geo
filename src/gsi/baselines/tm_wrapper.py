"""
Wrapper around Google's trimmed_match package.

trimmed_match provides a design-based confidence interval for iROAS in
paired geo experiments using trimmed match statistics. It is a fixed-time
method (no sequential validity).

The wrapper provides lazy import so the core gsi package works without
trimmed_match installed. If TM is needed but unavailable, a clear
ImportError is raised.
"""

import numpy as np


def is_tm_available() -> bool:
    """Check whether trimmed_match is installed."""
    try:
        import trimmed_match
        return True
    except ImportError:
        return False


def _get_tm():
    """Lazy import of trimmed_match with helpful error message."""
    try:
        import trimmed_match
        return trimmed_match
    except ImportError:
        raise ImportError(
            "The 'trimmed_match' package is required for TM baselines. "
            "Install it with: pip install geo_seq_inference[baselines]"
        )


def tm_ci(
    dY: np.ndarray,
    dS: np.ndarray,
    Z: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute trimmed-match confidence interval for iROAS.

    Wraps the trimmed_match package's confidence interval. This is a
    fixed-time method — no sequential/time-uniform guarantees.

    Args:
        dY: Cumulative response differences, shape (n_pairs,).
        dS: Cumulative spend differences, shape (n_pairs,).
        Z: Treatment assignment, shape (n_pairs,). +1 or -1.
        alpha: Significance level (default 0.05).

    Returns:
        (lower, upper) — 1-alpha confidence interval bounds.

    Raises:
        ImportError: If trimmed_match is not installed.
        NotImplementedError: If the TM API is not yet implemented.
    """
    tm = _get_tm()

    # ── Convert to trimmed_match's expected format ───────────────────────
    # trimmed_match expects:
    #   - response: per-geo response values
    #   - cost: per-geo spend/cost values
    #   - assignment: treatment indicator per geo
    #
    # Our data is per-pair differences. We need to convert back to per-geo.
    # For now, use the pair-level wrapper if available.

    # The trimmed_match package API may vary. Attempt to use the standard
    # TrimmedMatch class or report_pair_CI function.
    try:
        from trimmed_match.report import report_pair_CI
        # report_pair_CI expects (delta_response, delta_cost, assignment, ...)
        result = report_pair_CI(
            delta_response=dY,
            delta_cost=dS,
            assignment=(Z > 0).astype(int),
            alpha=alpha,
        )
        return float(result['ci_lower']), float(result['ci_upper'])
    except ImportError:
        pass

    # Fallback: try the TrimmedMatch class directly
    try:
        from trimmed_match.trimmed_match import TrimmedMatch
        tm_obj = TrimmedMatch(
            response=dY,
            cost=dS,
            assignment=(Z > 0).astype(int),
        )
        ci = tm_obj.confidence_interval(alpha=alpha)
        return float(ci[0]), float(ci[1])
    except Exception:
        pass

    # If none of the above worked, inform the user.
    raise NotImplementedError(
        "Could not interface with trimmed_match. The package API may have "
        "changed. Please check the trimmed_match documentation and update "
        "tm_wrapper.py accordingly."
    )
