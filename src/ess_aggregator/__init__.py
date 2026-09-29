"""ess_aggregator: merge ESS micro-data with country-year macro aggregates.

This package builds country-year (and country / pooled) aggregates of
European Social Survey (ESS) micro-data variables, suitable for merging back
onto macro-economic panel data for multilevel / hierarchical linear models or
panel regressions.
"""

from .aggregation import AggregationResult, aggregate_group
from .pipeline import run_pipeline

__all__ = ["AggregationResult", "aggregate_group", "run_pipeline"]

__version__ = "0.1.0"
