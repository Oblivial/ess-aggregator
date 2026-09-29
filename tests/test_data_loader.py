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
