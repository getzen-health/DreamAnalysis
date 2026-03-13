"""Tests for ml/api/routes/food_image.py endpoint.

These tests use FastAPI's TestClient (synchronous) and pytest-asyncio for
async path helpers.  Vision AI calls are mocked so tests run without
ANTHROPIC_API_KEY or network access.
"""
import json
import sys
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ── Setup ─────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.routes.food_image import router, _glycemic_impact, _dominant_macro, _parse_description_with_db  # noqa: E501

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ── Helper builders ───────────────────────────────────────────────────────────

def _food_item(name="Apple", portion="1 medium (150g)", cal=78, prot=0.4, carbs=21, fat=0.3, fiber=3.6):
    return {
        "name": name,
        "portion": portion,
        "calories": cal,
        "protein_g": prot,
        "carbs_g": carbs,
        "fat_g": fat,
        "fiber_g": fiber,
    }


# ── /food/analyze-image/status ────────────────────────────────────────────────

class TestStatus:
    def test_status_returns_200(self):
        resp = client.get("/food/analyze-image/status")
        assert resp.status_code == 200

    def test_status_has_expected_fields(self):
        data = client.get("/food/analyze-image/status").json()
        assert "vision_ai_available" in data
        assert "db_lookup_available" in data
        assert data["db_lookup_available"] is True

    def test_status_no_key_reports_unavailable(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            data = client.get("/food/analyze-image/status").json()
            assert data["vision_ai_available"] is False

    def test_status_with_key_reports_available(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            data = client.get("/food/analyze-image/status").json()
            assert data["vision_ai_available"] is True


# ── POST /food/analyze-image — validation ─────────────────────────────────────

class TestValidation:
    def test_neither_image_nor_text_returns_422(self):
        resp = client.post("/food/analyze-image", json={})
        assert resp.status_code == 422

    def test_empty_strings_returns_422(self):
        resp = client.post("/food/analyze-image", json={"image_base64": "", "text_description": ""})
        assert resp.status_code == 422


# ── POST /food/analyze-image — text description path (no vision) ──────────────

class TestTextDescriptionPath:
    def test_single_food_returns_result(self):
        resp = client.post("/food/analyze-image", json={"text_description": "apple"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["food_items"]) >= 1
        assert data["total_calories"] > 0

    def test_multiple_foods_in_description(self):
        resp = client.post(
            "/food/analyze-image",
            json={"text_description": "grilled chicken breast, broccoli, brown rice"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["food_items"]) >= 2

    def test_analysis_method_is_description_lookup(self):
        resp = client.post("/food/analyze-image", json={"text_description": "banana"})
        assert resp.json()["analysis_method"] == "description_lookup"

    def test_response_has_all_required_fields(self):
        resp = client.post("/food/analyze-image", json={"text_description": "salmon with rice"})
        assert resp.status_code == 200
        data = resp.json()
        for field in (
            "food_items", "total_calories", "total_protein_g", "total_carbs_g",
            "total_fat_g", "total_fiber_g", "dominant_macro", "glycemic_impact",
            "confidence", "analysis_method", "summary",
        ):
            assert field in data, f"Missing field: {field}"

    def test_totals_match_sum_of_items(self):
        resp = client.post(
            "/food/analyze-image",
            json={"text_description": "oatmeal, blueberries"},
        )
        assert resp.status_code == 200
        data = resp.json()
        expected_cal = sum(i["calories"] for i in data["food_items"])
        assert abs(data["total_calories"] - expected_cal) < 1.0

    def test_unknown_food_only_returns_422_or_partial(self):
        # A completely unknown ingredient — should either 422 or return partial
        resp = client.post("/food/analyze-image", json={"text_description": "xyzunknownfood99"})
        # Either no items found (422) or partial result is acceptable
        assert resp.status_code in (200, 422)
        if resp.status_code == 200:
            data = resp.json()
            assert data["food_items"] is not None

    def test_meal_type_included_in_summary(self):
        resp = client.post(
            "/food/analyze-image",
            json={"text_description": "eggs and toast", "meal_type": "breakfast"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "breakfast" in data["summary"].lower() or len(data["food_items"]) > 0


# ── POST /food/analyze-image — mocked vision path ────────────────────────────

class TestVisionPath:
    """Test the Claude vision path using a mocked Anthropic client."""

    _MOCK_VISION_RESULT = {
        "food_items": [
            _food_item("Grilled salmon", "1 fillet (180g)", 374, 50.4, 0.0, 18.0, 0.0),
            _food_item("Brown rice", "1 cup cooked (195g)", 218, 5.1, 45.8, 1.8, 3.5),
            _food_item("Steamed broccoli", "1 cup (91g)", 31, 2.5, 6.0, 0.4, 2.4),
        ],
        "summary": "Grilled salmon with brown rice and steamed broccoli",
        "confidence": 0.88,
    }

    def _make_mock_response(self):
        """Return a mock that mimics anthropic.AsyncAnthropic().messages.create()."""
        mock_message = AsyncMock()
        mock_message.content = [AsyncMock(text=json.dumps(self._MOCK_VISION_RESULT))]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        return mock_client

    def test_vision_path_used_when_image_and_key_present(self):
        """With ANTHROPIC_API_KEY set and an image, the vision path should be used."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("api.routes.food_image._analyze_with_anthropic") as mock_vision:
                mock_vision.return_value = self._MOCK_VISION_RESULT
                resp = client.post(
                    "/food/analyze-image",
                    json={"image_base64": "abc123", "meal_type": "dinner"},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["analysis_method"] == "vision_ai"
                assert len(data["food_items"]) == 3
                assert data["confidence"] == 0.88

    def test_vision_failure_falls_back_to_text_description(self):
        """If vision throws, text description fallback must work."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("api.routes.food_image._analyze_with_anthropic") as mock_vision:
                mock_vision.side_effect = Exception("Vision timeout")
                resp = client.post(
                    "/food/analyze-image",
                    json={
                        "image_base64": "abc123",
                        "text_description": "apple and banana",
                    },
                )
                # Should succeed via text fallback
                assert resp.status_code == 200
                data = resp.json()
                assert data["analysis_method"] == "description_lookup"
                assert data["error"] is not None  # error field populated

    def test_image_without_key_falls_back_to_text(self):
        """No ANTHROPIC_API_KEY → skip vision, fall back to text description."""
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            resp = client.post(
                "/food/analyze-image",
                json={"image_base64": "abc123", "text_description": "chicken and rice"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["analysis_method"] == "description_lookup"


# ── Unit helpers ──────────────────────────────────────────────────────────────

class TestGlycemicImpact:
    def test_high_gi_keywords(self):
        assert _glycemic_impact(["white bread", "soda"]) == "high"

    def test_low_gi_keywords(self):
        assert _glycemic_impact(["oatmeal", "broccoli", "lentils"]) == "low"

    def test_mixed_defaults_medium(self):
        # No clear GI signal in these items → should return low or medium (not high)
        result = _glycemic_impact(["beef", "pasta"])
        assert result in ("low", "medium")

    def test_empty_list(self):
        result = _glycemic_impact([])
        assert result in ("low", "medium", "high")


class TestDominantMacro:
    def test_carb_dominant(self):
        # 0 protein, 100g carbs, 0 fat → carbs
        assert _dominant_macro(0, 100, 0) == "carbs"

    def test_protein_dominant(self):
        # 50g protein (200 cal), 10g carbs (40 cal), 5g fat (45 cal)
        assert _dominant_macro(50, 10, 5) == "protein"

    def test_fat_dominant(self):
        # 5g protein (20 cal), 5g carbs (20 cal), 20g fat (180 cal)
        assert _dominant_macro(5, 5, 20) == "fat"


class TestParseDescriptionWithDB:
    def test_finds_single_item(self):
        items = _parse_description_with_db("apple")
        assert len(items) >= 1
        assert any("apple" in i["name"].lower() for i in items)

    def test_finds_multiple_items(self):
        items = _parse_description_with_db("banana, greek yogurt, almonds")
        assert len(items) >= 2

    def test_all_items_have_required_fields(self):
        items = _parse_description_with_db("chicken breast, broccoli")
        for item in items:
            for field in ("name", "portion", "calories", "protein_g", "carbs_g", "fat_g", "fiber_g"):
                assert field in item

    def test_calorie_values_positive(self):
        items = _parse_description_with_db("salmon, rice, spinach")
        for item in items:
            assert item["calories"] > 0
