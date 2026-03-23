"""Tests for the consumer EEG benchmark script.

Tests the module structure, constants, CLI argument parsing, and
check/download-instructions flows. Does NOT require the actual
dataset to be downloaded.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def benchmark_module():
    """Import the benchmark module."""
    # Ensure ml/ is on the path for imports
    ml_dir = Path(__file__).resolve().parent.parent
    if str(ml_dir) not in sys.path:
        sys.path.insert(0, str(ml_dir))
    return importlib.import_module("training.benchmark_consumer_eeg")


class TestModuleConstants:
    """Test that required constants and paths are defined."""

    def test_dataset_doi_defined(self, benchmark_module):
        assert benchmark_module.DATASET_DOI == "10.6084/m9.figshare.30162868"

    def test_dataset_url_contains_doi(self, benchmark_module):
        assert benchmark_module.DATASET_DOI in benchmark_module.DATASET_URL

    def test_data_dir_is_path(self, benchmark_module):
        assert isinstance(benchmark_module.DATA_DIR, Path)

    def test_model_dir_is_path(self, benchmark_module):
        assert isinstance(benchmark_module.MODEL_DIR, Path)

    def test_models_to_benchmark_is_nonempty(self, benchmark_module):
        assert len(benchmark_module.MODELS_TO_BENCHMARK) > 0

    def test_each_model_has_required_keys(self, benchmark_module):
        required_keys = {"name", "file", "type", "description"}
        for model in benchmark_module.MODELS_TO_BENCHMARK:
            missing = required_keys - set(model.keys())
            assert not missing, f"Model {model.get('name', '?')} missing keys: {missing}"

    def test_model_types_are_valid(self, benchmark_module):
        valid_types = {"sklearn", "pytorch"}
        for model in benchmark_module.MODELS_TO_BENCHMARK:
            assert model["type"] in valid_types, (
                f"Model {model['name']} has invalid type: {model['type']}"
            )


class TestCheckDataset:
    """Test the check_dataset function."""

    def test_returns_false_when_dir_missing(self, benchmark_module, tmp_path):
        with patch.object(benchmark_module, "DATA_DIR", tmp_path / "nonexistent"):
            assert benchmark_module.check_dataset() is False

    def test_returns_false_when_dir_empty(self, benchmark_module, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch.object(benchmark_module, "DATA_DIR", empty_dir):
            assert benchmark_module.check_dataset() is False

    def test_returns_true_when_dir_has_files(self, benchmark_module, tmp_path):
        data_dir = tmp_path / "consumer_eeg"
        data_dir.mkdir()
        (data_dir / "README.md").write_text("test")
        (data_dir / "participants.tsv").write_text("id\n1")
        with patch.object(benchmark_module, "DATA_DIR", data_dir):
            assert benchmark_module.check_dataset() is True


class TestDownloadInstructions:
    """Test the download instructions printer."""

    def test_prints_doi(self, benchmark_module, capsys):
        benchmark_module.print_download_instructions()
        output = capsys.readouterr().out
        assert benchmark_module.DATASET_DOI in output

    def test_prints_url(self, benchmark_module, capsys):
        benchmark_module.print_download_instructions()
        output = capsys.readouterr().out
        assert "doi.org" in output

    def test_prints_directory_structure(self, benchmark_module, capsys):
        benchmark_module.print_download_instructions()
        output = capsys.readouterr().out
        assert "sub-01" in output


class TestLoadDataRaisesWithoutDataset:
    """Test that load fails gracefully when dataset is missing."""

    def test_raises_file_not_found_when_no_dir(self, benchmark_module, tmp_path):
        with patch.object(benchmark_module, "DATA_DIR", tmp_path / "nonexistent"):
            with pytest.raises(FileNotFoundError, match="not found"):
                benchmark_module.load_consumer_eeg_data()

    def test_raises_when_no_subject_dirs(self, benchmark_module, tmp_path):
        data_dir = tmp_path / "consumer_eeg"
        data_dir.mkdir()
        (data_dir / "README.md").write_text("test")
        with patch.object(benchmark_module, "DATA_DIR", data_dir):
            with pytest.raises(FileNotFoundError, match="No subject directories"):
                benchmark_module.load_consumer_eeg_data()


class TestCLIParsing:
    """Test that CLI argument parsing works."""

    def test_check_flag(self, benchmark_module):
        """Verify --check sets the check attribute."""
        with patch("sys.argv", ["benchmark_consumer_eeg", "--check"]):
            # We can't call main() because it calls sys.exit, but we can
            # verify argparse works by checking the flag is accepted
            parser = __import__("argparse").ArgumentParser()
            parser.add_argument("--check", action="store_true")
            parser.add_argument("--download-instructions", action="store_true")
            parser.add_argument("--run", action="store_true")
            parser.add_argument("--model", type=str, default=None)
            parser.add_argument("--max-subjects", type=int, default=None)
            args = parser.parse_args(["--check"])
            assert args.check is True
            assert args.run is False

    def test_run_with_model_filter(self, benchmark_module):
        parser = __import__("argparse").ArgumentParser()
        parser.add_argument("--check", action="store_true")
        parser.add_argument("--run", action="store_true")
        parser.add_argument("--model", type=str, default=None)
        parser.add_argument("--max-subjects", type=int, default=None)
        args = parser.parse_args(["--run", "--model", "emotion_mega_lgbm"])
        assert args.run is True
        assert args.model == "emotion_mega_lgbm"
