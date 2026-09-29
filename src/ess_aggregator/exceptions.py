"""Custom exceptions used across ess_aggregator.

Keeping these distinct (instead of bare ``Exception``/``ValueError``) lets the
pipeline and CLI tell recoverable, per-unit problems (e.g. "this
country-year has too few respondents") apart from fatal, run-stopping
problems (e.g. "the ESS API is unreachable"), and log/handle them
accordingly.
"""

from __future__ import annotations


class ESSAggregatorError(Exception):
    """Base class for all errors raised by ess_aggregator."""


class VariableNotFoundError(ESSAggregatorError):
    """Raised when a requested variable does not exist in the ESS codebook."""


class RoundNotFoundError(ESSAggregatorError):
    """Raised when a requested ESS round label/DOI cannot be resolved."""


class DataLoadError(ESSAggregatorError):
    """Raised when downloading/parsing an ESS datafile fails."""


class InsufficientDataError(ESSAggregatorError):
    """Raised when a group has no usable (non-missing, positively weighted)
    observations at all, so no statistic can be computed."""
