"""Unit tests for ess_aggregator.data_loader, using the fake ESS client from
conftest.py (no network access)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ess_aggregator.data_loader import EssDataLoader
from ess_aggregator.exceptions import DataLoadError, RoundNotFoundError, VariableNotFoundError


class TestRoundResolution:
    def test_defaults_to_every_round_the_variable_is_in(self, fake_ess_two_rounds):
        loader = EssDataLoader(fake_ess_two_rounds)
        dois = loader.resolve_rounds_for_variable("stflife")
        assert dois == ["10.1/ess1", "10.1/ess2"]

    def test_restricting_to_a_subset(self, fake_ess_two_rounds):
        loader = EssDataLoader(fake_ess_two_rounds)
        dois = loader.resolve_rounds_for_variable("stflife", requested_rounds=["ESS1"])
        assert dois == ["10.1/ess1"]

    def test_unknown_variable_raises(self, fake_ess_two_rounds):
        loader = EssDataLoader(fake_ess_two_rounds)
        with pytest.raises(VariableNotFoundError):
            loader.resolve_rounds_for_variable("not_a_real_variable")

    def test_unknown_round_raises(self, fake_ess_two_rounds):
        loader = EssDataLoader(fake_ess_two_rounds)
        with pytest.raises(RoundNotFoundError):
            loader.resolve_rounds_for_variable("stflife", requested_rounds=["ESS99"])

    def test_round_without_variable_is_skipped_with_error_if_none_left(
        self, fake_ess_two_rounds
    ):
        # "cntry" is only registered without any `rounds`, so it has no round
        # membership at all -> DataLoadError.
        with pytest.raises(DataLoadError):
            EssDataLoader(fake_ess_two_rounds).resolve_rounds_for_variable("cntry")


class TestLoadVariableLong:
    def test_loads_and_concatenates_both_rounds(self, fake_ess_two_rounds):
        loader = EssDataLoader(fake_ess_two_rounds)
        long_df = loader.load_variable_long("stflife")
        assert set(long_df["essround"].unique()) == {1, 2}
        assert len(long_df) == 160  # 2 rounds * 80 respondents
        assert set(long_df.columns) == set(loader.LONG_COLUMNS)

    def test_country_is_decoded_to_full_name(self, fake_ess_two_rounds):
        loader = EssDataLoader(fake_ess_two_rounds)
        long_df = loader.load_variable_long("stflife")
        assert set(long_df["country"].unique()) == {"Germany", "France"}

    def test_country_filter(self, fake_ess_two_rounds):
        loader = EssDataLoader(fake_ess_two_rounds)
        long_df = loader.load_variable_long("stflife", countries=["DE"])
        assert set(long_df["cntry_code"].unique()) == {"DE"}

    def test_weight_column_is_anweight_when_present(self, fake_ess_two_rounds):
        loader = EssDataLoader(fake_ess_two_rounds)
        loader.load_variable_long("stflife")
        assert all(r.weight_column == "anweight" for r in loader.round_reports)

    def test_year_column_is_inwyys_when_present(self, fake_ess_two_rounds):
        loader = EssDataLoader(fake_ess_two_rounds)
        long_df = loader.load_variable_long("stflife")
        assert set(long_df["year"].unique()) == {2002, 2004}

    def test_missing_weight_column_falls_back_to_unweighted(self, fake_ess_two_rounds):
        fake_ess_two_rounds._round_data["10.1/ess1"] = fake_ess_two_rounds._round_data[
            "10.1/ess1"
        ].drop(columns=["anweight"])
        loader = EssDataLoader(fake_ess_two_rounds)
        long_df = loader.load_variable_long("stflife", requested_rounds=["ESS1"])
        assert (long_df["weight"] == 1.0).all()

    def test_broken_round_is_skipped_not_fatal(self, fake_ess_two_rounds, monkeypatch):
        original_load = fake_ess_two_rounds.load

        def flaky_load(doi, **kwargs):
            if doi == "10.1/ess1":
                raise RuntimeError("simulated network failure")
            return original_load(doi, **kwargs)

        monkeypatch.setattr(fake_ess_two_rounds, "load", flaky_load)
        loader = EssDataLoader(fake_ess_two_rounds)
        long_df = loader.load_variable_long("stflife")
        # Only round 2's 80 respondents should have made it through.
        assert len(long_df) == 80
        assert any(not r.ok for r in loader.round_reports)

    def test_all_rounds_failing_raises(self, fake_ess_two_rounds, monkeypatch):
        def always_fail(doi, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(fake_ess_two_rounds, "load", always_fail)
        loader = EssDataLoader(fake_ess_two_rounds)
        with pytest.raises(DataLoadError):
            loader.load_variable_long("stflife")


class TestLoadLocalCsv:
    def test_loads_local_csv_through_pyess_client(self, fake_ess_two_rounds, tmp_path):
        path = tmp_path / "ess.csv"
        fake_ess_two_rounds._round_data["10.1/ess1"].to_csv(path, index=False)
        loader = EssDataLoader(fake_ess_two_rounds, local_csv_path=path)

        long_df = loader.load_variable_long("stflife")

        assert len(long_df) == 80
        assert fake_ess_two_rounds.local_load_calls == [(path, ["stflife"], "pandas")]
        assert loader.round_reports[0].ok
        assert loader.round_reports[0].weight_column == "anweight"

    def test_uses_polars_engine_and_converts_frame(self, fake_ess_two_rounds, tmp_path):
        path = tmp_path / "ess.csv"
        fake_ess_two_rounds._round_data["10.1/ess1"].to_csv(path, index=False)
        loader = EssDataLoader(fake_ess_two_rounds, local_csv_path=path, engine="polars")

        long_df = loader.load_variable_long("stflife")

        assert len(long_df) == 80
        assert fake_ess_two_rounds.local_load_calls == [(path, ["stflife"], "polars")]
        assert set(long_df["country"]) == {"Germany", "France"}

    def test_reuses_local_frame_for_multiple_variables(self, fake_ess_two_rounds, tmp_path):
        path = tmp_path / "ess.csv"
        fake_ess_two_rounds._round_data["10.1/ess1"].to_csv(path, index=False)
        loader = EssDataLoader(fake_ess_two_rounds, local_csv_path=path)

        loader.load_variable_long("stflife")
        loader.load_variable_long("cntry")

        assert fake_ess_two_rounds.local_load_calls == [(path, ["stflife"], "pandas")]

    def test_recodes_missing_values_for_a_variable_loaded_after_the_first(
        self, fake_ess_two_rounds, tmp_path
    ):
        """Regression test: the local dataframe is only loaded (and
        recoded) once, for the first variable; a second variable accessed
        afterwards from the same cached dataframe must still get its own
        designated-missing codes recoded."""
        df = fake_ess_two_rounds._round_data["10.1/ess1"].copy()
        df["stfeco"] = [5.0] * 79 + [77.0]  # last respondent: "Refusal"
        fake_ess_two_rounds.codebook._variables["stfeco"] = type(
            fake_ess_two_rounds.codebook._variables["stflife"]
        )(missing_values={"77", "88", "99"})
        path = tmp_path / "ess.csv"
        df.to_csv(path, index=False)
        loader = EssDataLoader(fake_ess_two_rounds, local_csv_path=path)

        loader.load_variable_long("stflife")  # loads+caches the dataframe first
        long_df = loader.load_variable_long("stfeco")

        assert long_df["value"].isna().sum() == 1
        assert long_df["value"].max() == 5.0

    def test_filters_local_csv_by_requested_round(self, fake_ess_two_rounds, tmp_path):
        path = tmp_path / "ess.csv"
        pd.concat(fake_ess_two_rounds._round_data.values(), ignore_index=True).to_csv(
            path, index=False
        )
        loader = EssDataLoader(fake_ess_two_rounds, local_csv_path=path)

        long_df = loader.load_variable_long("stflife", requested_rounds=["ESS1"])

        assert len(long_df) == 80
        assert set(long_df["essround"]) == {1}

    def test_year_sentinel_missing_code_does_not_leak_into_other_rounds(
        self, fake_ess_two_rounds, tmp_path
    ):
        """Regression test: ESS uses a designated-missing sentinel (``9999``)
        for interview-year columns themselves. A round that only populates a
        lower-priority year candidate (e.g. ``inwyr``) while a higher-priority
        one (e.g. ``inwyys``) is coded with that sentinel used to make every
        such row resolve to the bogus year 9999 instead of falling back."""
        with_sentinel = fake_ess_two_rounds._round_data["10.1/ess1"].copy()
        with_sentinel["inwyys"] = 9999  # not available for this round
        with_sentinel["inwyr"] = 2002
        fake_ess_two_rounds.codebook._variables["inwyys"] = type(
            fake_ess_two_rounds.codebook._variables["stflife"]
        )(missing_values={"9999"})
        path = tmp_path / "ess.csv"
        with_sentinel.to_csv(path, index=False)
        loader = EssDataLoader(fake_ess_two_rounds, local_csv_path=path)

        long_df = loader.load_variable_long("stflife")

        assert set(long_df["year"].unique()) == {2002}

    def test_resolves_year_per_row_when_rounds_use_different_year_columns(
        self, fake_ess_two_rounds, tmp_path
    ):
        """Regression test: a merged local CSV stacking multiple ESS rounds
        can have early rounds populate only ``inwyr`` and later rounds only
        populate ``inwyys``. Picking a single year column for the whole file
        used to silently drop every row from whichever round didn't
        populate that column."""
        early_round = fake_ess_two_rounds._round_data["10.1/ess1"].copy()
        early_round["inwyys"] = np.nan
        early_round["inwyr"] = 2002
        later_round = fake_ess_two_rounds._round_data["10.1/ess2"].copy()
        later_round["inwyr"] = np.nan
        merged = pd.concat([early_round, later_round], ignore_index=True)
        path = tmp_path / "ess.csv"
        merged.to_csv(path, index=False)
        loader = EssDataLoader(fake_ess_two_rounds, local_csv_path=path)

        long_df = loader.load_variable_long("stflife")

        assert len(long_df) == 160
        assert set(long_df["year"].unique()) == {2002, 2004}
        assert loader.round_reports[0].year_column == "inwyys+inwyr"

    def test_resolves_weight_per_row_when_rounds_use_different_weight_columns(
        self, fake_ess_two_rounds, tmp_path
    ):
        """Regression test: a merged local CSV stacking multiple ESS rounds
        can have some rounds missing ``anweight`` entirely (only
        ``pspwght``/``dweight`` populated). Picking a single weight column
        for the whole file used to turn every row from such a round into an
        unusable (NaN-weighted, thus dropped) observation."""
        with_anweight = fake_ess_two_rounds._round_data["10.1/ess1"].copy()
        without_anweight = fake_ess_two_rounds._round_data["10.1/ess2"].copy()
        without_anweight["pspwght"] = without_anweight["anweight"]
        without_anweight = without_anweight.drop(columns=["anweight"])
        merged = pd.concat([with_anweight, without_anweight], ignore_index=True)
        path = tmp_path / "ess.csv"
        merged.to_csv(path, index=False)
        loader = EssDataLoader(fake_ess_two_rounds, local_csv_path=path)

        long_df = loader.load_variable_long("stflife")

        assert len(long_df) == 160
        assert not long_df["weight"].isna().any()
        assert loader.round_reports[0].weight_column == "anweight+pspwght"

    def test_round_filter_requires_essround_column(self, fake_ess_two_rounds, tmp_path):
        path = tmp_path / "ess.csv"
        fake_ess_two_rounds._round_data["10.1/ess1"].drop(columns=["essround"]).to_csv(
            path, index=False
        )
        loader = EssDataLoader(fake_ess_two_rounds, local_csv_path=path)

        with pytest.raises(DataLoadError, match="essround"):
            loader.load_variable_long("stflife", requested_rounds=["ESS1"])
