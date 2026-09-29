"""Pure, dependency-free statistical building blocks for country-year
aggregation of ESS micro-data.

Every function here operates on plain ``numpy`` arrays of ``values`` and
``weights`` (no pandas, no ESS-specific knowledge), which keeps the tricky
numerical business logic trivially unit-testable in isolation from data
loading/network concerns. See README.md for the mathematical background and
rationale behind each statistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import DEFAULT_MIN_EFFECTIVE_N
from .exceptions import InsufficientDataError


def _clean(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop respondents with a missing value and/or non-positive weight.

    Non-finite (NaN/inf) values and weights <= 0 cannot meaningfully
    contribute to a weighted statistic, so they are removed up front.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return values[mask], weights[mask]


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size: ``(sum(w))^2 / sum(w^2)``.

    Measures how many *equally-weighted* respondents would carry the same
    amount of statistical information as the actually-observed, unequally
    weighted sample. It is invariant to multiplying every weight by the same
    constant (e.g. a per-country ``pweight`` factor), so it is safe to use
    regardless of which of the weight candidates in ``config`` was selected.
    """
    if weights.size == 0:
        return 0.0
    sum_w = weights.sum()
    sum_w2 = np.square(weights).sum()
    if sum_w2 == 0:
        return 0.0
    return float(sum_w**2 / sum_w2)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total_w = weights.sum()
    if total_w == 0:
        return float("nan")
    return float(np.sum(values * weights) / total_w)


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted (population) variance around the weighted mean."""
    total_w = weights.sum()
    if total_w == 0:
        return float("nan")
    mean = weighted_mean(values, weights)
    return float(np.sum(weights * (values - mean) ** 2) / total_w)


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    variance = weighted_variance(values, weights)
    return float(np.sqrt(variance)) if variance == variance else float("nan")


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted percentile using the mid-point/Hazen interpolation method.

    ``q`` is a fraction in ``[0, 1]`` (e.g. ``0.5`` for the median). Values
    are sorted, and each observation ``i`` is assigned the cumulative weight
    fraction at its distribution *mid-point*::

        cw_i = (cumsum(w)_i - 0.5 * w_i) / sum(w)

    The requested quantile is then obtained by linear interpolation between
    the two bracketing ``cw_i`` (matching the common survey-statistics
    convention, e.g. R's ``Hmisc::wtd.quantile`` default). This degenerates
    to the standard unweighted percentile when all weights are equal.
    """
    if values.size == 0:
        return float("nan")
    if values.size == 1:
        return float(values[0])
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cum_w = np.cumsum(sorted_weights)
    total_w = cum_w[-1]
    if total_w == 0:
        return float("nan")
    mid_cum_w = (cum_w - 0.5 * sorted_weights) / total_w
    return float(np.interp(q, mid_cum_w, sorted_values))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return weighted_percentile(values, weights, 0.5)


def weighted_mode(values: np.ndarray, weights: np.ndarray) -> float:
    """The value with the largest total weight ("most typical" category).

    Most meaningful for discrete/categorical variables; for continuous
    variables it returns the single most heavily-weighted observed value
    (not a density-estimated mode), which is documented as a limitation in
    README.md.
    """
    if values.size == 0:
        return float("nan")
    unique_values, inverse = np.unique(values, return_inverse=True)
    weight_sums = np.zeros(unique_values.shape[0])
    np.add.at(weight_sums, inverse, weights)
    return float(unique_values[np.argmax(weight_sums)])


def weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Gini coefficient (relative inequality), in ``[0, 1]``.

    Uses the standard discrete Lorenz-curve/trapezoid formula for grouped
    data: sort ascending, compute the cumulative weight share ``W_i`` and
    cumulative value share ``L_i`` (the Lorenz curve ordinate) at each point,
    then::

        G = 1 - sum_i w_i_share * (L_{i-1} + L_i)

    which is twice the area *between* the Lorenz curve and the line of
    perfect equality. Undefined (returns ``NaN``) if the total value is zero
    or negative (Gini assumes a non-negative quantity, e.g. income).
    """
    if values.size == 0:
        return float("nan")
    if np.any(values < 0):
        # Gini is only defined for non-negative quantities; bail out rather
        # than silently returning a misleading number.
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    total_w = sorted_weights.sum()
    total_wx = np.sum(sorted_weights * sorted_values)
    if total_w == 0 or total_wx == 0:
        return float("nan")
    w_share = sorted_weights / total_w
    cum_value = np.cumsum(sorted_weights * sorted_values) / total_wx
    cum_value_prev = np.concatenate(([0.0], cum_value[:-1]))
    gini = 1.0 - np.sum(w_share * (cum_value_prev + cum_value))
    return float(gini)


def quantile_ratio(p_high: float, p_low: float) -> float:
    """Generic quantile ratio, e.g. P90/P10. NaN if the denominator is 0."""
    if p_low == 0 or not np.isfinite(p_low) or not np.isfinite(p_high):
        return float("nan")
    return float(p_high / p_low)


def palma_ratio(values: np.ndarray, weights: np.ndarray, p90: float, p40: float) -> float:
    """Palma ratio: share of total value held by the top 10% (values >= P90)
    divided by the share held by the bottom 40% (values <= P40).

    A simplified alternative to the Gini coefficient that focuses on the
    tails of the distribution, popular in inequality research because it is
    less sensitive to the (often noisy) middle of the distribution.
    """
    total_wx = np.sum(weights * values)
    if total_wx == 0 or not np.isfinite(p90) or not np.isfinite(p40):
        return float("nan")
    top_mask = values >= p90
    bottom_mask = values <= p40
    top_share = np.sum(weights[top_mask] * values[top_mask]) / total_wx
    bottom_share = np.sum(weights[bottom_mask] * values[bottom_mask]) / total_wx
    if bottom_share == 0:
        return float("nan")
    return float(top_share / bottom_share)


@dataclass
class AggregationResult:
    """All computed statistics for a single (variable, country, year) group,
    or a pooled ("All") variant thereof."""

    n_raw: int
    n_effective: float
    reliable: bool
    mean: float
    median: float
    mode: float
    std: float
    iqr: float
    p10: float
    p40: float
    p50: float
    p90: float
    p90_p10_ratio: float
    palma_ratio: float
    gini: float
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "n_raw": self.n_raw,
            "n_effective": round(self.n_effective, 2),
            "reliable": self.reliable,
            "mean": self.mean,
            "median": self.median,
            "mode": self.mode,
            "std": self.std,
            "iqr": self.iqr,
            "p10": self.p10,
            "p40": self.p40,
            "p50": self.p50,
            "p90": self.p90,
            "p90_p10_ratio": self.p90_p10_ratio,
            "palma_ratio": self.palma_ratio,
            "gini": self.gini,
            "warnings": "; ".join(self.warnings),
        }
        return d


def aggregate_group(
    values: np.ndarray,
    weights: np.ndarray,
    min_effective_n: int = DEFAULT_MIN_EFFECTIVE_N,
) -> AggregationResult:
    """Compute every statistic for one group of (already filtered to a single
    variable/country/year combination) respondent values + weights.

    Parameters
    ----------
    values, weights:
        Raw (not yet cleaned) arrays; missing values and non-positive
        weights are dropped internally.
    min_effective_n:
        See :data:`ess_aggregator.config.DEFAULT_MIN_EFFECTIVE_N`. Below this
        *effective* sample size, distribution-shape statistics (median,
        percentiles, IQR, Gini, quantile ratios) are considered unstable and
        reported as ``NaN`` (with a warning), while the mean/mode/std are
        still reported, since they are markedly less sensitive to small-N
        instability than quantile-based statistics.

    Raises
    ------
    InsufficientDataError
        If there is not a single usable observation at all (e.g. every
        respondent had a missing value, or a zero weight).
    """
    clean_values, clean_weights = _clean(values, weights)
    n_raw = clean_values.size
    if n_raw == 0:
        raise InsufficientDataError(
            "No usable (non-missing, positively-weighted) observations in this group."
        )

    n_eff = effective_sample_size(clean_weights)
    reliable = n_raw >= min_effective_n and n_eff >= min_effective_n
    warnings: list[str] = []
    if not reliable:
        warnings.append(
            f"Low sample size (n_raw={n_raw}, n_effective={n_eff:.1f} < "
            f"min_effective_n={min_effective_n}); distributional statistics "
            "(median/percentiles/Gini/IQR/ratios) suppressed as NaN."
        )

    mean = weighted_mean(clean_values, clean_weights)
    mode = weighted_mode(clean_values, clean_weights)
    std = weighted_std(clean_values, clean_weights)

    if reliable:
        p10 = weighted_percentile(clean_values, clean_weights, 0.10)
        p40 = weighted_percentile(clean_values, clean_weights, 0.40)
        p50 = weighted_percentile(clean_values, clean_weights, 0.50)
        p90 = weighted_percentile(clean_values, clean_weights, 0.90)
        p25 = weighted_percentile(clean_values, clean_weights, 0.25)
        p75 = weighted_percentile(clean_values, clean_weights, 0.75)
        iqr = p75 - p25
        median = p50
        ratio = quantile_ratio(p90, p10)
        palma = palma_ratio(clean_values, clean_weights, p90, p40)
        gini = weighted_gini(clean_values, clean_weights)
    else:
        p10 = p40 = p50 = p90 = iqr = float("nan")
        median = float("nan")
        ratio = float("nan")
        palma = float("nan")
        gini = float("nan")

    return AggregationResult(
        n_raw=n_raw,
        n_effective=n_eff,
        reliable=reliable,
        mean=mean,
        median=median,
        mode=mode,
        std=std,
        iqr=iqr,
        p10=p10,
        p40=p40,
        p50=p50,
        p90=p90,
        p90_p10_ratio=ratio,
        palma_ratio=palma,
        gini=gini,
        warnings=warnings,
    )
