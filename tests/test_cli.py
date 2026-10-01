"""Tests for the CLI argument parsing and end-to-end `main()` orchestration,
with the `pyess` import mocked out via sys.modules so no real network/package
is required."""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from ess_aggregator.cli import build_arg_parser, main
from ess_aggregator.config import DEFAULT_MIN_EFFECTIVE_N


class TestArgParser:
    def test_requires_at_least_one_variable(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args(["stflife"])
        assert args.variables == ["stflife"]
        assert args.rounds is None
        assert args.countries is None
        assert args.min_effective_n == DEFAULT_MIN_EFFECTIVE_N
        assert args.output == "ess_aggregated.csv"
        assert args.output_format is None
        assert args.input_csv is None
        assert args.engine == "pandas"
        assert args.recode_missing_values is True

    def test_multiple_variables_and_overrides(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "stflife",
                "happy",
                "--rounds",
                "ESS9",
                "ESS10",
                "--countries",
                "DE",
                "FR",
                "--min-n",
                "500",
                "--output",
                "out.csv",
                "--output-format",
                "parquet",
                "--input-csv",
                "ess.csv",
                "--engine",
                "polars",
                "--no-recode-missing",
            ]
        )
        assert args.variables == ["stflife", "happy"]
        assert args.rounds == ["ESS9", "ESS10"]
        assert args.countries == ["DE", "FR"]
        assert args.min_effective_n == 500
        assert args.output == "out.csv"
        assert args.output_format == "parquet"
        assert args.input_csv == "ess.csv"
        assert args.engine == "polars"
        assert args.recode_missing_values is False


@pytest.fixture
def mocked_pyess_module(monkeypatch, fake_ess_two_rounds):
    """Inject a fake `pyess` module into sys.modules so `cli.main`'s
    `from pyess import ESS` picks up our network-free fake client."""
    fake_module = types.ModuleType("pyess")
    fake_module.ESS = lambda *args, **kwargs: fake_ess_two_rounds
    monkeypatch.setitem(sys.modules, "pyess", fake_module)
    return fake_ess_two_rounds


class TestMainEndToEnd:
    def test_successful_run_writes_csv(self, mocked_pyess_module, tmp_path):
        output_path = tmp_path / "out.csv"
        log_path = tmp_path / "run.log"
        exit_code = main(
            [
                "stflife",
                "--min-n",
                "1",
                "--output",
                str(output_path),
                "--log-file",
                str(log_path),
            ]
        )
        assert exit_code == 0
        assert output_path.exists()
        assert log_path.exists()
        df = pd.read_csv(output_path)
        assert "GermanyAll" in set(df["unit"])

    def test_unknown_variable_returns_nonzero_exit(self, mocked_pyess_module, tmp_path):
        exit_code = main(
            [
                "not_a_real_variable",
                "--output",
                str(tmp_path / "out.csv"),
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )
        assert exit_code == 1
        assert not (tmp_path / "out.csv").exists()

    def test_local_csv_with_polars_engine(self, mocked_pyess_module, tmp_path):
        input_path = tmp_path / "ess.csv"
        mocked_pyess_module._round_data["10.1/ess1"].to_csv(input_path, index=False)
        output_path = tmp_path / "out.csv"

        exit_code = main(
            [
                "stflife",
                "--input-csv",
                str(input_path),
                "--engine",
                "polars",
                "--min-n",
                "1",
                "--output",
                str(output_path),
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )

        assert exit_code == 0
        assert mocked_pyess_module.local_load_calls == [
            (input_path, ["stflife"], "polars")
        ]
        assert "GermanyAll" in set(pd.read_csv(output_path)["unit"])

    def test_parquet_format_is_inferred_from_extension(
        self, mocked_pyess_module, tmp_path
    ):
        pytest.importorskip("pyarrow")
        output_path = tmp_path / "out.parquet"

        exit_code = main(
            [
                "stflife",
                "--min-n",
                "1",
                "--output",
                str(output_path),
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )

        assert exit_code == 0
        assert output_path.exists()
        assert "GermanyAll" in set(pd.read_parquet(output_path)["unit"])

    def test_polars_engine_requires_local_csv(self, tmp_path):
        with pytest.raises(SystemExit):
            main(
                [
                    "stflife",
                    "--engine",
                    "polars",
                    "--log-file",
                    str(tmp_path / "run.log"),
                ]
            )

    def test_missing_pyess_package_returns_exit_code_2(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "pyess", None)  # forces ImportError on `from pyess import ESS`
        exit_code = main(
            [
                "stflife",
                "--output",
                str(tmp_path / "out.csv"),
                "--log-file",
                str(tmp_path / "run.log"),
            ]
        )
        assert exit_code == 2
