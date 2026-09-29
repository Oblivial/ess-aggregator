"""Unit tests for the pure statistical building blocks in
ess_aggregator.aggregation. These use small, hand-computable examples so
expected values can be verified independently (e.g. brute-force Gini)."""

from __future__ import annotations

import numpy as np
import pytest

from ess_aggregator.aggregation import (
    aggregate_group,
    effective_sample_size,
    palma_ratio,
    quantile_ratio,
    weighted_gini,
    weighted_mean,
    weighted_median,
    weighted_mode,
    weighted_percentile,
    weighted_std,
)
from ess_aggregator.exceptions import InsufficientDataError


def brute_force_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """O(n^2) reference implementation of the weighted Gini coefficient,
    used only to cross-check the fast, sorted-based implementation."""
    num = 0.0
    for i in range(len(values)):
        for j in range(len(values)):
            num += weights[i] * weights[j] * abs(values[i] - values[j])
    denom = 2 * weights.sum() * np.sum(weights * values)
    return num / denom


class TestWeightedMean:
    def test_equal_weights_matches_plain_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.ones(4)
        assert weighted_mean(values, weights) == pytest.approx(2.5)

    def test_weights_shift_the_mean(self):
        values = np.array([0.0, 10.0])
        weights = np.array([3.0, 1.0])
        # 3 parts of 0, 1 part of 10 -> 10/4 = 2.5
        assert weighted_mean(values, weights) == pytest.approx(2.5)

    def test_zero_total_weight_is_nan(self):
        assert np.isnan(weighted_mean(np.array([1.0]), np.array([0.0])))

    def test_scale_invariance_to_constant_weight_multiplier(self):
        # A per-country-constant pweight factor must not change the mean
        # (see README "Gewichtung" section) - multiplying every weight by
        # the same constant is a no-op for the weighted mean.
        values = np.array([1.0, 5.0, 9.0])
        weights = np.array([1.0, 2.0, 3.0])
        base = weighted_mean(values, weights)
        scaled = weighted_mean(values, weights * 7.3)
        assert scaled == pytest.approx(base)


class TestWeightedPercentile:
    def test_median_of_odd_unweighted_series(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        assert weighted_median(values, weights) == pytest.approx(3.0)

    def test_single_value_returns_that_value(self):
        assert weighted_percentile(np.array([42.0]), np.array([1.0]), 0.9) == 42.0

    def test_empty_returns_nan(self):
        assert np.isnan(weighted_percentile(np.array([]), np.array([]), 0.5))

    def test_percentiles_are_monotonic(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=200)
        weights = rng.uniform(0.1, 2.0, size=200)
        p10 = weighted_percentile(values, weights, 0.10)
        p50 = weighted_percentile(values, weights, 0.50)
        p90 = weighted_percentile(values, weights, 0.90)
        assert p10 < p50 < p90

    def test_heavier_weight_pulls_percentile_towards_that_value(self):
        values = np.array([1.0, 2.0, 3.0])
        equal_weights = np.array([1.0, 1.0, 1.0])
        skewed_weights = np.array([1.0, 1.0, 100.0])
        assert weighted_median(values, skewed_weights) > weighted_median(
            values, equal_weights
        )


class TestWeightedStd:
    def test_matches_numpy_std_for_equal_weights(self):
        values = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        weights = np.ones(8)
        assert weighted_std(values, weights) == pytest.approx(np.std(values))


class TestWeightedMode:
    def test_returns_most_heavily_weighted_value(self):
        values = np.array([1.0, 2.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0, 5.0])
        assert weighted_mode(values, weights) == 3.0

    def test_unweighted_mode_is_most_frequent_value(self):
        values = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
        weights = np.ones(5)
        assert weighted_mode(values, weights) == 2.0


class TestEffectiveSampleSize:
    def test_equal_weights_gives_n(self):
        weights = np.ones(50)
        assert effective_sample_size(weights) == pytest.approx(50.0)

    def test_unequal_weights_reduce_effective_n(self):
        weights = np.array([1.0] * 49 + [1000.0])
        assert effective_sample_size(weights) < 50.0

    def test_invariant_to_constant_scaling(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        assert effective_sample_size(weights) == pytest.approx(
            effective_sample_size(weights * 100.0)
        )

    def test_empty_is_zero(self):
        assert effective_sample_size(np.array([])) == 0.0


class TestGini:
    @pytest.mark.parametrize(
        "values,weights",
        [
            (np.array([1.0, 1.0, 1.0, 1.0]), np.ones(4)),
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.ones(5)),
            (np.array([0.0, 10.0, 20.0, 100.0]), np.array([2.0, 1.0, 3.0, 1.0])),
        ],
    )
    def test_matches_brute_force_reference(self, values, weights):
        assert weighted_gini(values, weights) == pytest.approx(
            brute_force_gini(values, weights), abs=1e-9
        )

    def test_perfect_equality_is_zero(self):
        values = np.array([5.0, 5.0, 5.0, 5.0])
        weights = np.ones(4)
        assert weighted_gini(values, weights) == pytest.approx(0.0, abs=1e-9)

    def test_negative_values_are_undefined(self):
        assert np.isnan(weighted_gini(np.array([-1.0, 2.0]), np.ones(2)))


class TestRatiosAndPalma:
    def test_quantile_ratio_basic(self):
        assert quantile_ratio(90.0, 10.0) == pytest.approx(9.0)

    def test_quantile_ratio_zero_denominator_is_nan(self):
        assert np.isnan(quantile_ratio(50.0, 0.0))

    def test_palma_ratio_extreme_inequality(self):
        # 10 respondents: 1 has almost everything, 9 have almost nothing.
        values = np.array([0.01] * 9 + [100.0])
        weights = np.ones(10)
        p90 = weighted_percentile(values, weights, 0.90)
        p40 = weighted_percentile(values, weights, 0.40)
        ratio = palma_ratio(values, weights, p90, p40)
        assert ratio > 1.0


class TestAggregateGroup:
    def test_raises_on_all_missing(self):
        values = np.array([np.nan, np.nan])
        weights = np.array([1.0, 1.0])
        with pytest.raises(InsufficientDataError):
            aggregate_group(values, weights)

    def test_low_n_suppresses_distributional_stats_but_keeps_mean(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        result = aggregate_group(values, weights, min_effective_n=300)
        assert not result.reliable
        assert np.isnan(result.median)
        assert np.isnan(result.gini)
        assert not np.isnan(result.mean)
        assert result.warnings  # a warning must be recorded

    def test_high_n_computes_everything(self):
        rng = np.random.default_rng(42)
        values = rng.uniform(1, 10, size=500)
        weights = rng.uniform(0.5, 1.5, size=500)
        result = aggregate_group(values, weights, min_effective_n=300)
        assert result.reliable
        assert not np.isnan(result.median)
        assert not np.isnan(result.gini)
        assert result.n_raw == 500

    def test_missing_values_are_dropped_not_counted(self):
        values = np.array([1.0, 2.0, np.nan, 3.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        result = aggregate_group(values, weights, min_effective_n=1)
        assert result.n_raw == 3

    def test_zero_or_negative_weight_rows_are_dropped(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 0.0, -1.0])
        result = aggregate_group(values, weights, min_effective_n=1)
        assert result.n_raw == 1
