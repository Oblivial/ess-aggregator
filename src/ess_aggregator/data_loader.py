"""Data loading layer: wraps ``py-ess`` to build a single long/tidy
DataFrame (one row per respondent) for a requested variable, across every
ESS round it was collected in (or a user-restricted subset), with the
country/year/weight columns needed for aggregation already resolved.

This module is intentionally the only place that talks to ``py-ess``
directly (or to the network at all) - the rest of the pipeline works on
plain pandas DataFrames, which keeps it independently unit-testable with a
fake/mocked ESS client (see tests/conftest.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .config import (
    COUNTRY_COLUMN,
    POPULATION_WEIGHT_CANDIDATE,
    ROUND_COLUMN,
    WEIGHT_CANDIDATES,
    YEAR_COLUMN_CANDIDATES,
)
from .exceptions import DataLoadError, RoundNotFoundError, VariableNotFoundError

logger = logging.getLogger("ess_aggregator")


class SupportsCodebookVariable(Protocol):
    rounds: list[str]

    def label_for(self, value: Any) -> str | None: ...


class SupportsCodebook(Protocol):
    def get_variable(self, variable_id: str) -> SupportsCodebookVariable | None: ...

    def get_round(self, round_: str) -> Any | None: ...


class SupportsDataset(Protocol):
    dataframe: pd.DataFrame


class SupportsESSClient(Protocol):
    """The subset of ``pyess.ESS``'s interface this module relies on. Any
    object providing these members (real or fake/mocked) works."""

    codebook: SupportsCodebook

    def load(self, doi: str, **kwargs: Any) -> SupportsDataset: ...


@dataclass
class LoadedRoundInfo:
    """Small bookkeeping record of what happened loading one round, surfaced
    back to the pipeline for the processing log/summary."""

    round_label: str
    doi: str
    n_rows: int
    weight_column: str
    year_column: str | None
    ok: bool
    message: str = ""


class EssDataLoader:
    """Loads ESS variable data across rounds into a tidy long DataFrame."""

    LONG_COLUMNS = ["variable", "cntry_code", "country", "year", "essround", "value", "weight"]

    def __init__(
        self,
        ess_client: SupportsESSClient,
        recode_missing_values: bool = True,
        weight_candidates: tuple[str, ...] = WEIGHT_CANDIDATES,
        year_candidates: tuple[str, ...] = YEAR_COLUMN_CANDIDATES,
    ):
        self.ess = ess_client
        self.recode_missing_values = recode_missing_values
        self.weight_candidates = weight_candidates
        self.year_candidates = year_candidates
        self.round_reports: list[LoadedRoundInfo] = []

    # -- round resolution -------------------------------------------------
    def resolve_rounds_for_variable(
        self, variable_name: str, requested_rounds: list[str] | None = None
    ) -> list[str]:
        """Return the list of round DOIs to load for ``variable_name``.

        If ``requested_rounds`` is given, restricts to the intersection with
        the rounds the variable was actually collected in (logging a warning
        for any requested round that doesn't contain the variable). If
        omitted, every round the variable was ever collected in is used -
        this is the key convenience ``py-ess``'s variable-first indexing
        provides: the caller never has to manually enumerate rounds/DOIs.
        """
        variable = self.ess.codebook.get_variable(variable_name)
        if variable is None:
            raise VariableNotFoundError(
                f"Variable {variable_name!r} was not found in the ESS codebook."
            )
        available_dois = list(variable.rounds)
        if not available_dois:
            raise DataLoadError(f"Variable {variable_name!r} has no known round membership.")

        if requested_rounds is None:
            return available_dois

        resolved: list[str] = []
        for label in requested_rounds:
            round_obj = self.ess.codebook.get_round(label)
            if round_obj is None:
                raise RoundNotFoundError(f"Unknown ESS round {label!r}.")
            doi = round_obj.doi
            if doi not in available_dois:
                logger.warning(
                    "Round %s does not contain variable %r; skipping it for this variable.",
                    label,
                    variable_name,
                )
                continue
            resolved.append(doi)
        if not resolved:
            raise DataLoadError(
                f"None of the requested rounds {requested_rounds} contain variable "
                f"{variable_name!r}."
            )
        return resolved

    # -- weight / year column resolution -----------------------------------
    def _resolve_weight(self, df: pd.DataFrame) -> tuple[pd.Series, str]:
        """Pick the best available weight column.

        Prefers ``anweight`` (pre-combined design + population-size weight -
        see README.md "Gewichtung" for why this is safe to use for both
        single-country and pooled cross-country statistics). Falls back to
        deriving it as ``dweight_or_pspwght * pweight`` if only the raw
        components are present, then to an unweighted design weight, and
        finally to equal weights (1.0) with a loud warning - equal weights
        silently make every downstream statistic an *unweighted* one, which
        is why this is logged as a warning rather than happening quietly.
        """
        for candidate in self.weight_candidates:
            if candidate in df.columns:
                return df[candidate].astype(float), candidate

        design_weight_col = next(
            (c for c in ("pspwght", "dweight") if c in df.columns), None
        )
        if design_weight_col is not None and POPULATION_WEIGHT_CANDIDATE in df.columns:
            derived = df[design_weight_col].astype(float) * df[POPULATION_WEIGHT_CANDIDATE].astype(
                float
            )
            derived_name = f"{design_weight_col}*{POPULATION_WEIGHT_CANDIDATE}"
            logger.warning(
                "No pre-combined weight column found; derived %s from its components.",
                derived_name,
            )
            return derived, derived_name

        logger.warning(
            "No ESS design-weight column found (looked for %s); falling back to unweighted "
            "(weight=1.0) analysis. Results will not correct for sampling design.",
            self.weight_candidates,
        )
        return pd.Series(1.0, index=df.index), "none (unweighted)"

    def _resolve_year(self, df: pd.DataFrame) -> tuple[pd.Series, str | None]:
        for candidate in self.year_candidates:
            if candidate in df.columns:
                return pd.to_numeric(df[candidate], errors="coerce"), candidate
        logger.warning(
            "No interview-year column found (looked for %s); year-level aggregation is "
            "unavailable for this round.",
            self.year_candidates,
        )
        return pd.Series(np.nan, index=df.index), None

    def _decode_country(self, df: pd.DataFrame) -> pd.Series:
        country_variable = self.ess.codebook.get_variable(COUNTRY_COLUMN)
        codes = df[COUNTRY_COLUMN]
        if country_variable is None:
            return codes.astype(str)
        return codes.map(lambda code: country_variable.label_for(code) or str(code))

    # -- main entry point ---------------------------------------------------
    def load_variable_long(
        self,
        variable_name: str,
        requested_rounds: list[str] | None = None,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load ``variable_name`` across every applicable round and return a
        single tidy long DataFrame with columns
        ``["variable", "cntry_code", "country", "year", "essround", "value", "weight"]``.

        Rounds that fail to download/parse are logged and skipped rather
        than aborting the whole run (partial-success error handling), so one
        broken round doesn't prevent aggregating the rest.
        """
        dois = self.resolve_rounds_for_variable(variable_name, requested_rounds)
        frames: list[pd.DataFrame] = []
        countries_upper = {c.upper() for c in countries} if countries else None

        for doi in dois:
            round_obj = self.ess.codebook.get_datafile(doi) if hasattr(
                self.ess.codebook, "get_datafile"
            ) else None
            round_label = getattr(round_obj, "name", doi) if round_obj else doi
            try:
                dataset = self.ess.load(doi, recode_missing_values=self.recode_missing_values)
                df = dataset.dataframe
                if variable_name not in df.columns:
                    raise DataLoadError(
                        f"Round {round_label} ({doi}) does not actually contain column "
                        f"{variable_name!r} in its datafile (codebook/data mismatch)."
                    )
                if COUNTRY_COLUMN not in df.columns:
                    raise DataLoadError(
                        f"Round {round_label} ({doi}) is missing the required "
                        f"{COUNTRY_COLUMN!r} column."
                    )

                weight, weight_col = self._resolve_weight(df)
                year, year_col = self._resolve_year(df)
                country = self._decode_country(df)
                essround = (
                    pd.to_numeric(df[ROUND_COLUMN], errors="coerce")
                    if ROUND_COLUMN in df.columns
                    else pd.Series(np.nan, index=df.index)
                )

                long_df = pd.DataFrame(
                    {
                        "variable": variable_name,
                        "cntry_code": df[COUNTRY_COLUMN].astype(str),
                        "country": country,
                        "year": year,
                        "essround": essround,
                        "value": pd.to_numeric(df[variable_name], errors="coerce"),
                        "weight": weight,
                    }
                )

                if countries_upper is not None:
                    keep = long_df["cntry_code"].str.upper().isin(countries_upper) | long_df[
                        "country"
                    ].str.upper().isin(countries_upper)
                    long_df = long_df[keep]

                frames.append(long_df)
                self.round_reports.append(
                    LoadedRoundInfo(
                        round_label=round_label,
                        doi=doi,
                        n_rows=len(long_df),
                        weight_column=weight_col,
                        year_column=year_col,
                        ok=True,
                    )
                )
                logger.info(
                    "Loaded %s rows for variable=%r round=%s (weight=%s, year_col=%s)",
                    len(long_df),
                    variable_name,
                    round_label,
                    weight_col,
                    year_col,
                )
            except Exception as exc:  # noqa: BLE001 - deliberately broad: one bad
                # round must not abort the whole run; log and continue.
                logger.error(
                    "Failed to load variable=%r round=%s (%s): %s",
                    variable_name,
                    round_label,
                    doi,
                    exc,
                )
                self.round_reports.append(
                    LoadedRoundInfo(
                        round_label=round_label,
                        doi=doi,
                        n_rows=0,
                        weight_column="",
                        year_column=None,
                        ok=False,
                        message=str(exc),
                    )
                )

        if not frames:
            raise DataLoadError(
                f"Could not load any data at all for variable {variable_name!r}."
            )
        return pd.concat(frames, ignore_index=True)
