"""Shared pytest fixtures: a fake, network-free stand-in for the ``py-ess``
``ESS`` client, built to satisfy exactly the interface
:mod:`ess_aggregator.data_loader` relies on (see the ``Supports*`` Protocols
there). This lets the whole pipeline be tested deterministically and fast,
without any real ESS API access.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest


@dataclass
class FakeVariable:
    rounds: list[str] = field(default_factory=list)
    _labels: dict[str, str] = field(default_factory=dict)

    def label_for(self, value):
        return self._labels.get(str(value))


@dataclass
class FakeRound:
    doi: str
    name: str


@dataclass
class FakeDataset:
    dataframe: pd.DataFrame


class FakeCodebook:
    def __init__(self, variables: dict[str, FakeVariable], rounds: dict[str, FakeRound]):
        self._variables = variables
        # rounds keyed both by short label (e.g. "ESS1") and by DOI, mirroring
        # the real Codebook.get_round's dual lookup behaviour.
        self._rounds_by_label = rounds
        self._rounds_by_doi = {r.doi: r for r in rounds.values()}

    def get_variable(self, variable_id):
        return self._variables.get(variable_id)

    def get_round(self, round_):
        return self._rounds_by_label.get(round_) or self._rounds_by_doi.get(round_)

    def get_datafile(self, doi):
        return self._rounds_by_doi.get(doi)


class FakeESS:
    def __init__(self, codebook: FakeCodebook, round_data: dict[str, pd.DataFrame]):
        self.codebook = codebook
        self._round_data = round_data
        self.load_calls: list[str] = []

    def load(self, doi, **kwargs):
        self.load_calls.append(doi)
        if doi not in self._round_data:
            raise RuntimeError(f"No fake data registered for DOI {doi!r}")
        return FakeDataset(dataframe=self._round_data[doi].copy())


COUNTRY_LABELS = {"DE": "Germany", "FR": "France"}


def _make_round_df(essround: int, year: int, seed: int) -> pd.DataFrame:
    """40 DE + 40 FR synthetic respondents for one round."""
    rng = np.random.default_rng(seed)
    n_per_country = 40
    countries = ["DE"] * n_per_country + ["FR"] * n_per_country
    # stflife: 0-10 life-satisfaction-like scale; Germany centered higher.
    stflife = np.concatenate(
        [
            rng.integers(4, 11, size=n_per_country),
            rng.integers(2, 9, size=n_per_country),
        ]
    ).astype(float)
    anweight = rng.uniform(0.5, 1.5, size=2 * n_per_country)
    return pd.DataFrame(
        {
            "cntry": countries,
            "essround": essround,
            "inwyys": year,
            "anweight": anweight,
            "stflife": stflife,
        }
    )


@pytest.fixture
def fake_ess_two_rounds() -> FakeESS:
    """A fake ESS client with 2 rounds of synthetic 'stflife' data for DE/FR."""
    variables = {
        "stflife": FakeVariable(rounds=["10.1/ess1", "10.1/ess2"]),
        "cntry": FakeVariable(_labels=COUNTRY_LABELS),
    }
    rounds = {
        "ESS1": FakeRound(doi="10.1/ess1", name="ESS1 - 2002"),
        "ESS2": FakeRound(doi="10.1/ess2", name="ESS2 - 2004"),
    }
    codebook = FakeCodebook(variables, rounds)
    round_data = {
        "10.1/ess1": _make_round_df(essround=1, year=2002, seed=1),
        "10.1/ess2": _make_round_df(essround=2, year=2004, seed=2),
    }
    return FakeESS(codebook, round_data)
