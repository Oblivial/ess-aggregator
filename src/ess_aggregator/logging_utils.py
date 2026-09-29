"""Logging setup for ess_aggregator.

Provides a single :func:`configure_logging` entry point used by both the CLI
and tests, writing structured, timestamped records to a log file (as
required for auditing processing/errors) while also echoing to the console.
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

logger = logging.getLogger("ess_aggregator")


def configure_logging(log_file: str | None = None, verbose: bool = False) -> logging.Logger:
    """Configure the package-wide logger.

    Parameters
    ----------
    log_file:
        Path to write a persistent processing/error log to. If ``None``, only
        console logging is configured.
    verbose:
        If ``True``, sets the console handler to ``DEBUG`` level; otherwise
        ``INFO``. The file handler always logs at ``DEBUG`` level so the log
        file retains full detail regardless of console verbosity.
    """
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
