"""Command-line interface for ess_aggregator.

Usage
-----
    python -m ess_aggregator VARIABLE [VARIABLE ...] [options]

See README.md for a full walkthrough and the underlying business logic.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import logging
import sys

from .config import DEFAULT_MIN_EFFECTIVE_N
from .env import load_dotenv
from .exceptions import ESSAggregatorError
from .logging_utils import configure_logging
from .output import write_csv
from .pipeline import run_pipeline

logger_name = "ess_aggregator"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ess_aggregator",
        description=(
            "Aggregate European Social Survey (ESS) micro-data variables into "
            "country-year (and country/pooled) macro units, for merging onto "
            "macro-economic panel data (multilevel models / panel regressions)."
        ),
    )
    parser.add_argument(
        "variables",
        nargs="+",
        help="One or more ESS variable names to aggregate, e.g. 'stflife happy'.",
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        default=None,
        metavar="ROUND",
        help=(
            "Restrict to specific ESS rounds (e.g. 'ESS9 ESS10 ESS11'). "
            "Default: every round the variable was collected in."
        ),
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        metavar="COUNTRY",
        help=(
            "Restrict to specific countries, by ISO alpha-2 code or name "
            "(e.g. 'DE FR' or 'Germany France'). Default: all countries."
        ),
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=DEFAULT_MIN_EFFECTIVE_N,
        dest="min_effective_n",
        help=(
            "Minimum effective (Kish-weighted) sample size below which "
            f"distributional statistics are suppressed as NaN (default: {DEFAULT_MIN_EFFECTIVE_N})."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="ess_aggregated.csv",
        help="Path to the output CSV file (default: ess_aggregated.csv).",
    )
    parser.add_argument(
        "--log-file",
        default="ess_aggregator.log",
        help="Path to the processing/error log file (default: ess_aggregator.log).",
    )
    parser.add_argument(
        "--no-recode-missing",
        action="store_false",
        dest="recode_missing_values",
        help="Disable asking the ESS API to recode designated missing values to NaN.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level console logging (the log file is always DEBUG-level).",
    )
    return parser


def _log_pyess_provenance(logger: logging.Logger) -> None:
    """Log which ``py-ess`` install is actually in use (version + location).

    A stale or non-editable ``py-ess`` install silently shadowing a local
    development checkout is a common source of confusing runtime errors
    (e.g. an outdated version rejecting API requests). Logging this
    up-front makes that kind of drift visible instead of a mystery.

    Purely diagnostic: any failure to determine version/location is logged
    and swallowed rather than aborting the run.
    """
    try:
        version = importlib.metadata.version("py-ess")
    except importlib.metadata.PackageNotFoundError:
        logger.warning("py-ess version metadata not found (unusual for a working install).")
        return

    try:
        spec = importlib.util.find_spec("pyess")
        source_location = spec.origin if spec else "<unknown>"
    except (ImportError, ValueError):
        source_location = "<unknown>"
    logger.info("Using py-ess %s (source: %s)", version, source_location)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logger = configure_logging(args.log_file, verbose=args.verbose)
    logger.info("Starting ess_aggregator for variables=%s", args.variables)

    try:
        # Imported lazily so `--help` works without py-ess (and its network
        # dependencies) installed.
        from pyess import ESS

        _log_pyess_provenance(logger)
        ess_client = ESS()
        result_df = run_pipeline(
            ess_client,
            variables=args.variables,
            requested_rounds=args.rounds,
            countries=args.countries,
            min_effective_n=args.min_effective_n,
            recode_missing_values=args.recode_missing_values,
        )
    except ESSAggregatorError as exc:
        logger.error("Fatal error: %s", exc)
        return 1
    except ImportError:
        logger.error(
            "The 'pyess' package is required. Install it with: "
            "pip install git+https://github.com/Oblivial/py-ess.git@v0.1.0b2"
        )
        return 2
    except Exception:  # noqa: BLE001 - top-level safety net, always log the traceback
        logger.exception("Unexpected error while running the pipeline.")
        return 1

    if result_df.empty:
        logger.warning("No rows were produced; nothing written to %s", args.output)
        return 1

    written_path = write_csv(result_df, args.output)
    unreliable = int((~result_df["reliable"]).sum())
    logger.info(
        "Done. %d aggregate rows written to %s (%d flagged as low-N/unreliable).",
        len(result_df),
        written_path,
        unreliable,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
