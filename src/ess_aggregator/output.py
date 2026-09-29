"""CSV output writer for the aggregated result table."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("ess_aggregator")


def write_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write the aggregated DataFrame to ``output_path`` as CSV (UTF-8,
    comma-separated), creating parent directories if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %d rows to %s", len(df), path)
    return path
