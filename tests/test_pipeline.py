"""Unit tests for ess_aggregator.pipeline: unit-label construction, the
country/year/pooled group structure, and the Mundlak decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ess_aggregator.pipeline import aggregate_variable, aggregate_variables, run_pipeline


def _toy_long_df() -> pd.DataFrame:
    """A tiny, hand-constructed long DataFrame: Germany in 2 years, France in
    1 year, all with equal weights and enough rows to pass a low min_n."""
    rows = []
    # Germany 2001: values 2,4,6 -> mean 4
    for v in (2.0, 4.0, 6.0):
        rows.append({"variable": "x", "country": "Germany", "year": 2001, "value": v, "weight": 1.0})
    # Germany 2003: values 6,8,10 -> mean 8
    for v in (6.0, 8.0, 10.0):
        rows.append({"variable": "x", "country": "Germany", "year": 2003, "value": v, "weight": 1.0})
    # France 2001: values 1,2,3 -> mean 2
    for v in (1.0, 2.0, 3.0):
        rows.append({"variable": "x", "country": "France", "year": 2001, "value": v, "weight": 1.0})
    return pd.DataFrame(rows)


class TestAggregateVariableUnits:
    def test_produces_all_four_unit_types(self):
        result = aggregate_variable(_toy_long_df(), "x", min_effective_n=1)
        assert set(result["unit_type"]) == {
            "country_year",
            "country_all",
            "year_all",
            "grand_all",
        }

    def test_unit_labels_follow_country_year_convention(self):
        result = aggregate_variable(_toy_long_df(), "x", min_effective_n=1)
        units = set(result["unit"])
        assert {"Germany2001", "Germany2003", "GermanyAll", "France2001", "FranceAll"} <= units
        assert "All2001" in units
        assert "AllAll" in units

    def test_country_all_pools_across_years(self):
        result = aggregate_variable(_toy_long_df(), "x", min_effective_n=1)
        germany_all = result.loc[result["unit"] == "GermanyAll"].iloc[0]
        assert germany_all["mean"] == pytest.approx(np.mean([2, 4, 6, 6, 8, 10]))

    def test_year_all_pools_across_countries(self):
        result = aggregate_variable(_toy_long_df(), "x", min_effective_n=1)
        all_2001 = result.loc[result["unit"] == "All2001"].iloc[0]
        assert all_2001["mean"] == pytest.approx(np.mean([2, 4, 6, 1, 2, 3]))

    def test_grand_all_pools_everything(self):
        result = aggregate_variable(_toy_long_df(), "x", min_effective_n=1)
        grand = result.loc[result["unit"] == "AllAll"].iloc[0]
        assert grand["mean"] == pytest.approx(np.mean([2, 4, 6, 6, 8, 10, 1, 2, 3]))


class TestMundlakDecomposition:
    def test_country_mean_over_years_is_simple_average_of_yearly_means(self):
        result = aggregate_variable(_toy_long_df(), "x", min_effective_n=1)
        germany_2001 = result.loc[result["unit"] == "Germany2001"].iloc[0]
        germany_2003 = result.loc[result["unit"] == "Germany2003"].iloc[0]
        # Germany's yearly means are 4 and 8 -> simple average is 6, NOT the
        # pooled (weighted-by-N) GermanyAll mean of the raw respondents.
        assert germany_2001["country_mean_over_years"] == pytest.approx(6.0)
        assert germany_2003["country_mean_over_years"] == pytest.approx(6.0)

    def test_within_country_deviation_sums_to_zero_per_country(self):
        result = aggregate_variable(_toy_long_df(), "x", min_effective_n=1)
        germany_rows = result.loc[
            (result["unit_type"] == "country_year") & (result["country"] == "Germany")
        ]
        assert germany_rows["within_country_deviation"].sum() == pytest.approx(0.0)

    def test_pooled_units_have_no_mundlak_values(self):
        result = aggregate_variable(_toy_long_df(), "x", min_effective_n=1)
        pooled = result.loc[result["unit_type"] != "country_year"]
        assert pooled["country_mean_over_years"].isna().all()
        assert pooled["within_country_deviation"].isna().all()

    def test_single_year_country_has_zero_deviation(self):
        # France only has one year of data, so its within-deviation is 0 and
        # its country mean equals that single year's mean.
        result = aggregate_variable(_toy_long_df(), "x", min_effective_n=1)
        france_2001 = result.loc[result["unit"] == "France2001"].iloc[0]
        assert france_2001["within_country_deviation"] == pytest.approx(0.0)
        assert france_2001["country_mean_over_years"] == pytest.approx(2.0)


class TestAggregateVariablesMulti:
    def test_multiple_variables_are_concatenated(self):
        df = _toy_long_df()
        df2 = df.copy()
        df2["variable"] = "y"
        combined = pd.concat([df, df2], ignore_index=True)
        result = aggregate_variables(combined, ["x", "y"], min_effective_n=1)
        assert set(result["variable"]) == {"x", "y"}


class TestRunPipelineEndToEnd:
    def test_end_to_end_with_fake_ess_client(self, fake_ess_two_rounds):
        result = run_pipeline(
            fake_ess_two_rounds,
            variables=["stflife"],
            min_effective_n=1,
        )
        assert not result.empty
        assert "GermanyAll" in set(result["unit"])
        assert "FranceAll" in set(result["unit"])
        # min_effective_n=1 means even the 40-respondent country-year groups
        # are 'reliable' and get full distributional statistics.
        row = result.loc[result["unit"] == "Germany2002"].iloc[0]
        assert row["reliable"]
        assert not np.isnan(row["gini"])

    def test_high_min_n_flags_everything_unreliable(self, fake_ess_two_rounds):
        result = run_pipeline(
            fake_ess_two_rounds,
            variables=["stflife"],
            min_effective_n=1000,
        )
        assert not result["reliable"].any()
        assert result["gini"].isna().all()
