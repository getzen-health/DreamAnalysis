"""Tests for ml/data/nutrition_db.py."""
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db():
    """Import the nutrition DB module once for all tests."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.nutrition_db import lookup, lookup_with_portion, portion_size_to_grams, FOODS, all_food_names
    return {
        "lookup": lookup,
        "lookup_with_portion": lookup_with_portion,
        "portion_size_to_grams": portion_size_to_grams,
        "FOODS": FOODS,
        "all_food_names": all_food_names,
    }


# ── DB completeness ───────────────────────────────────────────────────────────

class TestDBCompleteness:
    def test_at_least_200_foods(self, db):
        assert len(db["FOODS"]) >= 200, f"Expected >=200 foods, got {len(db['FOODS'])}"

    def test_all_entries_have_required_fields(self, db):
        required = {"name", "calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "category"}
        for name, entry in db["FOODS"].items():
            for field in required:
                assert field in entry, f"'{name}' missing field '{field}'"

    def test_calorie_values_plausible(self, db):
        for name, entry in db["FOODS"].items():
            cal = entry["calories"]
            assert 0 <= cal <= 1000, f"'{name}' has implausible calories={cal}"

    def test_macro_values_non_negative(self, db):
        for name, entry in db["FOODS"].items():
            for macro in ("protein_g", "carbs_g", "fat_g", "fiber_g"):
                assert entry[macro] >= 0, f"'{name}' has negative {macro}={entry[macro]}"

    def test_categories_are_valid(self, db):
        valid = {"fruit", "vegetable", "grain", "protein", "dairy", "legume", "nut", "fat",
                 "snack", "beverage", "condiment", "baked", "prepared"}
        for name, entry in db["FOODS"].items():
            assert entry["category"] in valid, f"'{name}' has unknown category '{entry['category']}'"

    def test_covers_major_food_groups(self, db):
        categories = {e["category"] for e in db["FOODS"].values()}
        for expected in ("fruit", "vegetable", "grain", "protein", "dairy", "legume", "nut"):
            assert expected in categories, f"Category '{expected}' not represented in DB"


# ── lookup() ─────────────────────────────────────────────────────────────────

class TestLookup:
    def test_exact_match_case_insensitive(self, db):
        result = db["lookup"]("Apple")
        assert result is not None
        assert result["name"] == "Apple"

    def test_exact_match_lowercase(self, db):
        result = db["lookup"]("apple")
        assert result is not None
        assert result["name"] == "Apple"

    def test_fuzzy_match_partial_name(self, db):
        result = db["lookup"]("chiken breast")  # typo
        assert result is not None, "Should fuzzy-match 'chiken breast' to 'Chicken breast (cooked)'"

    def test_known_foods_found(self, db):
        must_find = [
            "banana", "broccoli", "white rice (cooked)", "salmon (cooked)",
            "almonds", "greek yogurt (plain, full fat)", "lentils (cooked)",
            "olive oil", "avocado",
        ]
        for food in must_find:
            result = db["lookup"](food)
            assert result is not None, f"'{food}' not found in DB"

    def test_unknown_food_returns_none(self, db):
        result = db["lookup"]("xyzunknownfood12345")
        assert result is None

    def test_returns_copy_not_reference(self, db):
        a = db["lookup"]("apple")
        b = db["lookup"]("apple")
        assert a is not b, "lookup() should return independent dicts"
        a["calories"] = 9999
        c = db["lookup"]("apple")
        assert c["calories"] != 9999, "Modifying result should not affect DB"


# ── lookup_with_portion() ─────────────────────────────────────────────────────

class TestLookupWithPortion:
    def test_scales_correctly(self, db):
        # Banana is 89 kcal/100g; 200g should be ~178 kcal
        result = db["lookup_with_portion"]("banana", 200)
        assert result is not None
        assert abs(result["calories"] - 178) < 2, f"Expected ~178 kcal, got {result['calories']}"

    def test_portion_g_field_present(self, db):
        result = db["lookup_with_portion"]("apple", 150)
        assert result["portion_g"] == 150

    def test_all_macro_fields_scaled(self, db):
        result = db["lookup_with_portion"]("almonds", 30)
        assert result is not None
        for field in ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g"):
            assert field in result

    def test_100g_matches_base_entry(self, db):
        base = db["lookup"]("broccoli")
        scaled = db["lookup_with_portion"]("broccoli", 100)
        assert abs(scaled["calories"] - base["calories"]) < 0.1
        assert abs(scaled["protein_g"] - base["protein_g"]) < 0.1

    def test_unknown_food_returns_none(self, db):
        result = db["lookup_with_portion"]("xyzunknown", 100)
        assert result is None


# ── portion_size_to_grams() ───────────────────────────────────────────────────

class TestPortionSizeToGrams:
    def test_gram_suffix(self, db):
        assert db["portion_size_to_grams"]("150g") == 150.0
        assert db["portion_size_to_grams"]("75 g") == 75.0
        assert db["portion_size_to_grams"]("200 grams") == 200.0

    def test_size_words(self, db):
        assert db["portion_size_to_grams"]("small") == 80.0
        assert db["portion_size_to_grams"]("medium") == 120.0
        assert db["portion_size_to_grams"]("large") == 180.0

    def test_cup(self, db):
        assert db["portion_size_to_grams"]("cup") == 240.0

    def test_1_cup(self, db):
        assert db["portion_size_to_grams"]("1 cup") == 240.0

    def test_half_cup(self, db):
        assert db["portion_size_to_grams"]("half cup") == 120.0

    def test_tablespoon(self, db):
        assert db["portion_size_to_grams"]("tablespoon") == 15.0

    def test_2_tablespoons(self, db):
        result = db["portion_size_to_grams"]("2 tablespoons")
        assert abs(result - 30.0) < 0.1

    def test_ounce(self, db):
        result = db["portion_size_to_grams"]("oz")
        assert abs(result - 28.3) < 0.5

    def test_unknown_returns_default(self, db):
        result = db["portion_size_to_grams"]("a handful of mystery")
        # Should return 100 g default
        assert result == 100.0


# ── all_food_names() ──────────────────────────────────────────────────────────

class TestAllFoodNames:
    def test_returns_list(self, db):
        names = db["all_food_names"]()
        assert isinstance(names, list)
        assert len(names) >= 200

    def test_sorted(self, db):
        names = db["all_food_names"]()
        assert names == sorted(names)

    def test_contains_known_foods(self, db):
        names_lower = {n.lower() for n in db["all_food_names"]()}
        for food in ("apple", "broccoli", "salmon (cooked)", "almonds"):
            assert food in names_lower, f"'{food}' not in all_food_names()"
