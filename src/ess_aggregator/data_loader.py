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
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pyess import recode_missing_values

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
    missing_values: set[str]

    def label_for(self, value: Any) -> str | None: ...


class SupportsCodebook(Protocol):
    def get_variable(self, variable_id: str) -> SupportsCodebookVariable | None: ...

    def get_round(self, round_: str) -> Any | None: ...


class SupportsDataset(Protocol):
    dataframe: Any


class SupportsESSClient(Protocol):
    """The subset of ``pyess.ESS``'s interface this module relies on. Any
    object providing these members (real or fake/mocked) works."""

    codebook: SupportsCodebook

    def load(self, doi: str, **kwargs: Any) -> SupportsDataset: ...

    def load_local_csv_by_round(
        self,
        path: str | Path,
        variables: list[str] | None = None,
        engine: str = "pandas",
        recode_missing_values: bool = True,
    ) -> dict[str, SupportsDataset]: ...


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
        local_csv_path: str | Path | None = None,
        engine: str = "pandas",
    ):
        self.ess = ess_client
        self.recode_missing_values = recode_missing_values
        self.weight_candidates = weight_candidates
        self.year_candidates = year_candidates
        self.local_csv_path = Path(local_csv_path) if local_csv_path is not None else None
        self.engine = engine
        # Local CSVs are split into one dataframe per round (DOI) the first
        # time any variable is requested, then reused for every later call -
        # mirroring how the API naturally hands back one round per request.
        self._local_round_dataframes: dict[str, pd.DataFrame] | None = None
        # Which (doi, variable) columns have already been missing-value-recoded
        # within `_local_round_dataframes`; recoding is applied lazily, per
        # round and per variable, the first time that column is accessed.
        self._recoded_local_columns: set[tuple[str, str]] = set()
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
        """Pick the best available weight column for this round.

        Prefers ``anweight`` (pre-combined design + population-size weight -
        see README.md "Gewichtung" for why this is safe to use for both
        single-country and pooled cross-country statistics), falling back to
        ``pspwght``/``dweight`` (standalone design weight, no population
        scaling), and finally a derived ``dweight_or_pspwght * pweight`` if
        the raw components are present. This only ever sees a single round's
        data (``py-ess`` hands back one round per DOI, for both the API and
        local-CSV sources - see ``load_local_csv_by_round``), so picking one
        column for the whole frame is correct: every row in a round either
        has the chosen column populated or none do. Falls back to equal
        weights (1.0) with a loud warning if nothing usable is found - equal
        weights silently make every downstream statistic an *unweighted*
        one, which is why this is logged as a warning rather than happening
        quietly.
        """
        for candidate in self.weight_candidates:
            if candidate in df.columns:
                return df[candidate].astype(float), candidate

        design_weight_col = next(
            (c for c in ("pspwght", "dweight") if c in df.columns), None
        )
        if design_weight_col is not None and POPULATION_WEIGHT_CANDIDATE in df.columns:
            derived_name = f"{design_weight_col}*{POPULATION_WEIGHT_CANDIDATE}"
            logger.warning(
                "No pre-combined weight column found; derived %s from its components.",
                derived_name,
            )
            weight = df[design_weight_col].astype(float) * df[POPULATION_WEIGHT_CANDIDATE].astype(
                float
            )
            return weight, derived_name

        logger.warning(
            "No ESS design-weight column found (looked for %s); falling back to "
            "unweighted (weight=1.0). Results will not correct for sampling design.",
            self.weight_candidates,
        )
        return pd.Series(1.0, index=df.index, dtype="float64"), "none (unweighted)"

    def _missing_codes_for(self, column: str) -> set[float]:
        """Numeric designated-missing codes (e.g. ``9999``) registered in the
        codebook for ``column``, if any (see ``Variable.missing_values``)."""
        variable = self.ess.codebook.get_variable(column)
        if variable is None:
            return set()
        codes: set[float] = set()
        for code in variable.missing_values:
            try:
                codes.add(float(code))
            except (TypeError, ValueError):
                continue
        return codes

    def _resolve_year(self, df: pd.DataFrame) -> tuple[pd.Series, str | None]:
        """Resolve this round's interview year from the best available
        candidate column.

        Different ESS rounds use different year columns (e.g. early rounds
        use ``inwyr`` while later ones use ``inwyys``), but - as with
        ``_resolve_weight`` - this only ever sees a single round's data, so
        picking the first candidate column present is correct. ESS uses a
        designated-missing sentinel (e.g. ``9999``) for "not
        applicable"/"not available" in year columns themselves, so the
        chosen column's own designated-missing codes are recoded to NaN
        before being returned.
        """
        for candidate in self.year_candidates:
            if candidate not in df.columns:
                continue
            year = pd.to_numeric(df[candidate], errors="coerce")
            year = year.mask(year.isin(self._missing_codes_for(candidate)))
            return year, candidate
        logger.warning(
            "None of the candidate year columns %s were present for this round; year is "
            "unavailable for this round.",
            self.year_candidates,
        )
        return pd.Series(np.nan, index=df.index, dtype="float64"), None

    def _decode_country(self, df: pd.DataFrame) -> pd.Series:
        country_variable = self.ess.codebook.get_variable(COUNTRY_COLUMN)
        codes = df[COUNTRY_COLUMN]
        if country_variable is None:
            return codes.astype(str)
        return codes.map(lambda code: country_variable.label_for(code) or str(code))

    @staticmethod
    def _to_pandas(dataframe: Any) -> pd.DataFrame:
        """Normalize py-ess pandas or Polars input for the aggregation code."""
        if isinstance(dataframe, pd.DataFrame):
            return dataframe
        to_pandas = getattr(dataframe, "to_pandas", None)
        if callable(to_pandas):
            return to_pandas()
        raise DataLoadError(
            f"Unsupported dataframe type {type(dataframe).__name__!r}; "
            "expected pandas or Polars."
        )

    def _prepare_long_dataframe(
        self,
        dataframe: Any,
        variable_name: str,
        countries: list[str] | None,
        source_label: str,
    ) -> tuple[pd.DataFrame, str, str | None]:
        """Validate and normalize one round's dataset into the pipeline
        format. Only ever receives a single round's data - ``py-ess`` hands
        back one dataframe per DOI for both the API and local-CSV sources
        (see ``load_local_csv_by_round``) - so no round filtering happens
        here; ``resolve_rounds_for_variable`` already restricted which
        DOIs/rounds are loaded in the first place.
        """
        df = self._to_pandas(dataframe)
        if variable_name not in df.columns:
            raise DataLoadError(
                f"{source_label} does not contain requested variable {variable_name!r}."
            )
        if COUNTRY_COLUMN not in df.columns:
            raise DataLoadError(
                f"{source_label} is missing the required {COUNTRY_COLUMN!r} column."
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
        countries_upper = {c.upper() for c in countries} if countries else None
        if countries_upper is not None:
            keep = long_df["cntry_code"].str.upper().isin(countries_upper) | long_df[
                "country"
            ].str.upper().isin(countries_upper)
            long_df = long_df[keep]
        logger.info(
            "Loaded %d rows for variable=%r from %s (weight=%s, year_col=%s)",
            len(long_df),
            variable_name,
            source_label,
            weight_col,
            year_col,
        )
        return long_df, weight_col, year_col

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

        Works identically whether ``self.local_csv_path`` is set or not: in
        both cases this loops once per round DOI and loads exactly that
        round's data via ``_load_round_dataframe`` - for the API that's a
        direct per-round request, for a local CSV it's a lazily-cached,
        already-split-by-round dataframe (see ``py-ess``'s
        ``load_local_csv_by_round``). Rounds that fail to load/parse are
        logged and skipped rather than aborting the whole run
        (partial-success error handling), so one broken round doesn't
        prevent aggregating the rest.
        """
        dois = self.resolve_rounds_for_variable(variable_name, requested_rounds)
        if self.local_csv_path is not None:
            # Fail fast on file-level structural problems (e.g. a missing
            # round-identifying column affects every round identically, so
            # there's no point retrying it once per round below).
            self._ensure_local_round_dataframes()
        frames: list[pd.DataFrame] = []

        for doi in dois:
            round_obj = self.ess.codebook.get_datafile(doi) if hasattr(
                self.ess.codebook, "get_datafile"
            ) else None
            round_label = getattr(round_obj, "name", doi) if round_obj else doi
            source_label = (
                f"local CSV {str(self.local_csv_path)!r} (round {round_label})"
                if self.local_csv_path is not None
                else f"Round {round_label} ({doi})"
            )
            try:
                dataframe = self._load_round_dataframe(doi, variable_name)
                long_df, weight_col, year_col = self._prepare_long_dataframe(
                    dataframe,
                    variable_name,
                    countries,
                    source_label,
                )

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

    # -- per-round dataframe loading (API or local CSV) ----------------------
    def _load_round_dataframe(self, doi: str, variable_name: str) -> pd.DataFrame:
        """Return one round's dataframe, dispatching to the API or the
        cached local-CSV round split depending on ``self.local_csv_path``."""
        if self.local_csv_path is not None:
            return self._load_local_round_dataframe(doi, variable_name)
        dataset = self.ess.load(doi, recode_missing_values=self.recode_missing_values)
        return self._to_pandas(dataset.dataframe)

    def _ensure_local_round_dataframes(self) -> dict[str, pd.DataFrame]:
        """Lazily split the local CSV into one dataframe per round (DOI),
        caching the result across calls so the (potentially huge) file is
        only read once, regardless of how many variables are requested.
        Missing-value recoding is deliberately *not* done here - it happens
        lazily, per round and per variable, in ``_load_local_round_dataframe``.
        """
        if self._local_round_dataframes is None:
            try:
                datasets = self.ess.load_local_csv_by_round(
                    self.local_csv_path,
                    engine=self.engine,
                    recode_missing_values=False,
                )
            except DataLoadError:
                raise
            except Exception as exc:
                raise DataLoadError(
                    f"Could not load local CSV {str(self.local_csv_path)!r}: {exc}"
                ) from exc
            self._local_round_dataframes = {
                doi: self._to_pandas(dataset.dataframe) for doi, dataset in datasets.items()
            }
        return self._local_round_dataframes

    def _load_local_round_dataframe(self, doi: str, variable_name: str) -> pd.DataFrame:
        source_label = f"local CSV {str(self.local_csv_path)!r}"
        round_dataframes = self._ensure_local_round_dataframes()
        if doi not in round_dataframes:
            raise DataLoadError(f"{source_label} has no data for round {doi!r}.")
        df = round_dataframes[doi]
        if variable_name not in df.columns:
            raise DataLoadError(
                f"{source_label} does not contain requested variable {variable_name!r} "
                f"for round {doi!r}."
            )
        cache_key = (doi, variable_name)
        if self.recode_missing_values and cache_key not in self._recoded_local_columns:
            df = recode_missing_values(df, self.ess.codebook, columns=[variable_name])
            round_dataframes[doi] = df
            self._recoded_local_columns.add(cache_key)
        return df

