"""Tests for the training sync route — corrections from Supabase for retraining."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure ml/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes.training_sync import (
    _append_corrections,
    _corrections_path,
    _load_local_corrections,
    CORRECTIONS_DIR,
    SyncResult,
    TrainingStatus,
)


@pytest.fixture(autouse=True)
def tmp_corrections_dir(tmp_path, monkeypatch):
    """Redirect corrections storage to a temp directory."""
    import api.routes.training_sync as mod
    monkeypatch.setattr(mod, "CORRECTIONS_DIR", tmp_path / "corrections")
    return tmp_path / "corrections"


@pytest.fixture
def sample_corrections():
    return [
        {
            "id": "aaa",
            "user_id": "user1",
            "predicted_emotion": "happy",
            "corrected_emotion": "sad",
            "created_at": "2026-03-23T10:00:00Z",
        },
        {
            "id": "bbb",
            "user_id": "user1",
            "predicted_emotion": "neutral",
            "corrected_emotion": "angry",
            "created_at": "2026-03-23T11:00:00Z",
        },
    ]


class TestLocalCorrections:
    def test_load_empty(self):
        """No file exists — returns empty list."""
        result = _load_local_corrections("nonexistent_user")
        assert result == []

    def test_append_and_load(self, sample_corrections):
        added = _append_corrections("user1", sample_corrections)
        assert added == 2

        loaded = _load_local_corrections("user1")
        assert len(loaded) == 2
        assert loaded[0]["predicted_emotion"] == "happy"
        assert loaded[1]["corrected_emotion"] == "angry"

    def test_deduplication(self, sample_corrections):
        """Same created_at timestamp should not be doubled."""
        _append_corrections("user1", sample_corrections)
        # Append the same records again
        added = _append_corrections("user1", sample_corrections)
        assert added == 0

        loaded = _load_local_corrections("user1")
        assert len(loaded) == 2  # still 2, not 4


class TestSyncEndpoint:
    @pytest.mark.asyncio
    async def test_sync_returns_count(self, sample_corrections):
        """Sync endpoint returns the number of synced corrections."""
        with patch(
            "api.routes.training_sync._get_supabase_corrections",
            return_value=sample_corrections,
        ):
            from api.routes.training_sync import sync_corrections

            result = await sync_corrections("user1")
            assert result.synced == 2
            assert result.new_corrections == 2
            assert result.total_corrections == 2

    @pytest.mark.asyncio
    async def test_no_supabase_configured(self):
        """Handles gracefully when Supabase is not configured."""
        with patch(
            "api.routes.training_sync._get_supabase_corrections",
            return_value=[],
        ):
            from api.routes.training_sync import sync_corrections

            result = await sync_corrections("user1")
            assert result.synced == 0
            assert "not be configured" in result.message.lower() or "no supabase" in result.message.lower()

    @pytest.mark.asyncio
    async def test_retrain_triggered_after_5(self, sample_corrections):
        """Retrain triggered when 5+ new corrections."""
        corrections_5 = sample_corrections + [
            {**sample_corrections[0], "id": f"c{i}", "created_at": f"2026-03-23T12:0{i}:00Z"}
            for i in range(3)
        ]
        assert len(corrections_5) == 5

        mock_retrain = MagicMock(return_value={"trained": True, "n_samples": 5})
        with patch(
            "api.routes.training_sync._get_supabase_corrections",
            return_value=corrections_5,
        ), patch(
            "training.auto_retrainer.retrain_personal_model",
            mock_retrain,
        ):
            from api.routes.training_sync import sync_corrections

            result = await sync_corrections("user1")
            assert result.new_corrections == 5
            assert result.retrain_triggered is True


class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_returns_stats(self, sample_corrections):
        """Status endpoint shows training data stats."""
        _append_corrections("user1", sample_corrections)

        with patch("api.routes.training_sync.get_supabase", return_value=None):
            # Need to patch the import inside the function
            with patch(
                "api.routes.training_sync._get_supabase_corrections",
                return_value=[],
            ):
                pass

        from api.routes.training_sync import training_status

        # Patch supabase to return None (not configured)
        with patch("lib.supabase_client.get_supabase", return_value=None):
            result = await training_status("user1")
            assert result.user_id == "user1"
            assert result.local_corrections == 2
            assert result.retrain_available is False  # < 5
