"""Central configuration constants for ess_aggregator.

Collecting these in one place makes the underlying business-logic decisions
(which weight to use, which columns identify country/year, the minimum
reliable sample size, ...) explicit and easy to audit/override, instead of
being scattered as magic strings/numbers through the pipeline.
"""

from __future__ import annotations

#: ESS design-weight columns, in order of preference for a *pooled*
#: (cross-country) analysis. ``anweight`` (= post-stratification weight *
#: population-size weight, further calibrated by ESS) is the recommended
#: default: see README.md, section "Gewichtung", for the full rationale.
WEIGHT_CANDIDATES: tuple[str, ...] = ("anweight", "pspwght", "dweight")

#: Fallback: population-size weight, only meaningful when combined with a
#: design weight. Used to *derive* an anweight-equivalent if the ESS API
#: response is missing the pre-computed ``anweight`` column outright.
POPULATION_WEIGHT_CANDIDATE = "pweight"

#: Respondent-level columns that identify the interview year, in order of
#: preference. ``inwyys``/``inwyye`` (start/end of interview, year) are more
#: precise than the older, single-value ``inwyr`` (used in early rounds), and
#: both are more precise than assuming every respondent in a round was
#: interviewed in that round's nominal year (fieldwork frequently spans a
#: year boundary).
YEAR_COLUMN_CANDIDATES: tuple[str, ...] = ("inwyys", "inwyr", "inwyye")

#: Column identifying the respondent's country (ISO 3166-1 alpha-2 code).
COUNTRY_COLUMN = "cntry"

#: Column identifying the ESS round number (1, 2, 3, ...).
ROUND_COLUMN = "essround"

#: Below this *effective* (Kish-weighted) sample size, distributional
#: statistics (median, percentiles, Gini, IQR, quantile ratios) are
#: considered unstable and are reported as ``NaN`` with a logged warning,
#: per the task's "N < 300-500" guidance. Configurable via the CLI.
DEFAULT_MIN_EFFECTIVE_N = 300

#: Placeholder used for the pooled-across-countries unit label, and for the
#: pooled-across-years suffix, e.g. ``GermanyAll``, ``All2001``, ``AllAll``.
ALL_LABEL = "All"

#: Codes that the ESS API's ``recodeMissingValues=true`` option (used by
#: default when downloading data) converts designated missing values (e.g.
#: "Refusal", "Don't know", "Not applicable") into. pandas already reads
#: these as NaN, but kept here for documentation purposes / defensive reuse.
MISSING_VALUE_SENTINEL = float("nan")
