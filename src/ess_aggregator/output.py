"""Output writers for the aggregated result table."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("ess_aggregator")


def write_output(
    df: pd.DataFrame,
    output_path: str | Path,
    output_format: str | None = None,
) -> Path:
    """Write aggregated results as CSV or Parquet.

    If ``output_format`` is omitted, the format is inferred from the file
    suffix; paths without a ``.parquet``/``.pq`` suffix default to CSV.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_format = output_format or (
        "parquet" if path.suffix.lower() in {".parquet", ".pq"} else "csv"
    )
    if file_format == "csv":
        df.to_csv(path, index=False, encoding="utf-8")
    elif file_format == "parquet":
        parquet_df = df.copy()
        if "year" in parquet_df.columns:
            # The output includes both numeric years and the "All" label.
            # A single string type keeps that dimension Parquet-compatible.
            parquet_df["year"] = parquet_df["year"].astype("string")
        parquet_df.to_parquet(path, index=False)
    else:
        raise ValueError(
            f"Unsupported output format {file_format!r}; expected 'csv' or 'parquet'."
        )
    logger.info("Wrote %d rows to %s as %s", len(df), path, file_format)
    return path


def write_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write the aggregated DataFrame to ``output_path`` as UTF-8 CSV."""
    return write_output(df, output_path, output_format="csv")
