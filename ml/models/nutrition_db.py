"""Nutritional database — per-100g macronutrient values for 200+ common foods.

All values are approximate averages from USDA FoodData Central and are
provided for informational purposes, not medical nutrition advice.

Usage
-----
    from data.nutrition_db import lookup, lookup_with_portion, FOODS

    # Exact or fuzzy lookup by food name
    entry = lookup("apple")
    # → {"name": "Apple", "calories": 52, "protein_g": 0.3, "carbs_g": 14.0,
    #    "fat_g": 0.2, "fiber_g": 2.4, "category": "fruit"}

    # Scale to a given portion weight
    entry = lookup_with_portion("chicken breast", grams=150)
"""

from __future__ import annotations

import difflib
from typing import Dict, List, Optional

# ── Data ──────────────────────────────────────────────────────────────────────
# Format: name → {calories, protein_g, carbs_g, fat_g, fiber_g, category}
# All values per 100 g (or 100 ml for liquids).
# "calories" = kcal.

_FOODS_RAW: List[Dict] = [
    # ── Fruits ────────────────────────────────────────────────────────────────
    {"name": "Apple", "calories": 52, "protein_g": 0.3, "carbs_g": 14.0, "fat_g": 0.2, "fiber_g": 2.4, "category": "fruit"},
    {"name": "Banana", "calories": 89, "protein_g": 1.1, "carbs_g": 23.0, "fat_g": 0.3, "fiber_g": 2.6, "category": "fruit"},
    {"name": "Orange", "calories": 47, "protein_g": 0.9, "carbs_g": 12.0, "fat_g": 0.1, "fiber_g": 2.4, "category": "fruit"},
    {"name": "Grapes", "calories": 69, "protein_g": 0.7, "carbs_g": 18.0, "fat_g": 0.2, "fiber_g": 0.9, "category": "fruit"},
    {"name": "Strawberries", "calories": 33, "protein_g": 0.7, "carbs_g": 8.0, "fat_g": 0.3, "fiber_g": 2.0, "category": "fruit"},
    {"name": "Blueberries", "calories": 57, "protein_g": 0.7, "carbs_g": 14.5, "fat_g": 0.3, "fiber_g": 2.4, "category": "fruit"},
    {"name": "Raspberries", "calories": 53, "protein_g": 1.2, "carbs_g": 12.0, "fat_g": 0.7, "fiber_g": 6.5, "category": "fruit"},
    {"name": "Mango", "calories": 60, "protein_g": 0.8, "carbs_g": 15.0, "fat_g": 0.4, "fiber_g": 1.6, "category": "fruit"},
    {"name": "Pineapple", "calories": 50, "protein_g": 0.5, "carbs_g": 13.1, "fat_g": 0.1, "fiber_g": 1.4, "category": "fruit"},
    {"name": "Watermelon", "calories": 30, "protein_g": 0.6, "carbs_g": 7.6, "fat_g": 0.2, "fiber_g": 0.4, "category": "fruit"},
    {"name": "Peach", "calories": 39, "protein_g": 0.9, "carbs_g": 10.0, "fat_g": 0.3, "fiber_g": 1.5, "category": "fruit"},
    {"name": "Pear", "calories": 57, "protein_g": 0.4, "carbs_g": 15.2, "fat_g": 0.1, "fiber_g": 3.1, "category": "fruit"},
    {"name": "Kiwi", "calories": 61, "protein_g": 1.1, "carbs_g": 14.7, "fat_g": 0.5, "fiber_g": 3.0, "category": "fruit"},
    {"name": "Avocado", "calories": 160, "protein_g": 2.0, "carbs_g": 9.0, "fat_g": 15.0, "fiber_g": 7.0, "category": "fruit"},
    {"name": "Coconut meat", "calories": 354, "protein_g": 3.3, "carbs_g": 15.2, "fat_g": 33.5, "fiber_g": 9.0, "category": "fruit"},
    {"name": "Lemon", "calories": 29, "protein_g": 1.1, "carbs_g": 9.3, "fat_g": 0.3, "fiber_g": 2.8, "category": "fruit"},
    {"name": "Lime", "calories": 30, "protein_g": 0.7, "carbs_g": 10.5, "fat_g": 0.2, "fiber_g": 2.8, "category": "fruit"},
    {"name": "Cherries", "calories": 63, "protein_g": 1.1, "carbs_g": 16.0, "fat_g": 0.2, "fiber_g": 2.1, "category": "fruit"},
    {"name": "Pomegranate", "calories": 83, "protein_g": 1.7, "carbs_g": 18.7, "fat_g": 1.2, "fiber_g": 4.0, "category": "fruit"},
    {"name": "Cantaloupe", "calories": 34, "protein_g": 0.8, "carbs_g": 8.2, "fat_g": 0.2, "fiber_g": 0.9, "category": "fruit"},

    # ── Vegetables ───────────────────────────────────────────────────────────
    {"name": "Broccoli", "calories": 34, "protein_g": 2.8, "carbs_g": 7.0, "fat_g": 0.4, "fiber_g": 2.6, "category": "vegetable"},
    {"name": "Spinach", "calories": 23, "protein_g": 2.9, "carbs_g": 3.6, "fat_g": 0.4, "fiber_g": 2.2, "category": "vegetable"},
    {"name": "Kale", "calories": 49, "protein_g": 4.3, "carbs_g": 9.0, "fat_g": 0.9, "fiber_g": 3.6, "category": "vegetable"},
    {"name": "Carrot", "calories": 41, "protein_g": 0.9, "carbs_g": 10.0, "fat_g": 0.2, "fiber_g": 2.8, "category": "vegetable"},
    {"name": "Cucumber", "calories": 15, "protein_g": 0.7, "carbs_g": 3.6, "fat_g": 0.1, "fiber_g": 0.5, "category": "vegetable"},
    {"name": "Tomato", "calories": 18, "protein_g": 0.9, "carbs_g": 3.9, "fat_g": 0.2, "fiber_g": 1.2, "category": "vegetable"},
    {"name": "Bell pepper", "calories": 31, "protein_g": 1.0, "carbs_g": 6.0, "fat_g": 0.3, "fiber_g": 2.1, "category": "vegetable"},
    {"name": "Onion", "calories": 40, "protein_g": 1.1, "carbs_g": 9.3, "fat_g": 0.1, "fiber_g": 1.7, "category": "vegetable"},
    {"name": "Garlic", "calories": 149, "protein_g": 6.4, "carbs_g": 33.1, "fat_g": 0.5, "fiber_g": 2.1, "category": "vegetable"},
    {"name": "Zucchini", "calories": 17, "protein_g": 1.2, "carbs_g": 3.1, "fat_g": 0.3, "fiber_g": 1.0, "category": "vegetable"},
    {"name": "Eggplant", "calories": 25, "protein_g": 1.0, "carbs_g": 5.9, "fat_g": 0.2, "fiber_g": 3.0, "category": "vegetable"},
    {"name": "Cauliflower", "calories": 25, "protein_g": 1.9, "carbs_g": 5.0, "fat_g": 0.3, "fiber_g": 2.0, "category": "vegetable"},
    {"name": "Brussels sprouts", "calories": 43, "protein_g": 3.4, "carbs_g": 9.0, "fat_g": 0.3, "fiber_g": 3.8, "category": "vegetable"},
    {"name": "Asparagus", "calories": 20, "protein_g": 2.2, "carbs_g": 3.9, "fat_g": 0.1, "fiber_g": 2.1, "category": "vegetable"},
    {"name": "Green beans", "calories": 31, "protein_g": 1.8, "carbs_g": 7.0, "fat_g": 0.1, "fiber_g": 2.7, "category": "vegetable"},
    {"name": "Celery", "calories": 16, "protein_g": 0.7, "carbs_g": 3.0, "fat_g": 0.2, "fiber_g": 1.6, "category": "vegetable"},
    {"name": "Lettuce (romaine)", "calories": 17, "protein_g": 1.2, "carbs_g": 3.3, "fat_g": 0.3, "fiber_g": 2.1, "category": "vegetable"},
    {"name": "Mushrooms", "calories": 22, "protein_g": 3.1, "carbs_g": 3.3, "fat_g": 0.3, "fiber_g": 1.0, "category": "vegetable"},
    {"name": "Sweet corn", "calories": 86, "protein_g": 3.2, "carbs_g": 18.7, "fat_g": 1.2, "fiber_g": 2.0, "category": "vegetable"},
    {"name": "Peas", "calories": 81, "protein_g": 5.4, "carbs_g": 14.5, "fat_g": 0.4, "fiber_g": 5.1, "category": "vegetable"},
    {"name": "Beet", "calories": 43, "protein_g": 1.6, "carbs_g": 10.0, "fat_g": 0.2, "fiber_g": 2.8, "category": "vegetable"},
    {"name": "Sweet potato", "calories": 86, "protein_g": 1.6, "carbs_g": 20.1, "fat_g": 0.1, "fiber_g": 3.0, "category": "vegetable"},
    {"name": "Potato", "calories": 77, "protein_g": 2.0, "carbs_g": 17.5, "fat_g": 0.1, "fiber_g": 2.2, "category": "vegetable"},
    {"name": "Radish", "calories": 16, "protein_g": 0.7, "carbs_g": 3.4, "fat_g": 0.1, "fiber_g": 1.6, "category": "vegetable"},
    {"name": "Leek", "calories": 61, "protein_g": 1.5, "carbs_g": 14.2, "fat_g": 0.3, "fiber_g": 1.8, "category": "vegetable"},

    # ── Grains & carbohydrates ────────────────────────────────────────────────
    {"name": "White rice (cooked)", "calories": 130, "protein_g": 2.7, "carbs_g": 28.2, "fat_g": 0.3, "fiber_g": 0.4, "category": "grain"},
    {"name": "Brown rice (cooked)", "calories": 112, "protein_g": 2.6, "carbs_g": 23.5, "fat_g": 0.9, "fiber_g": 1.8, "category": "grain"},
    {"name": "Oats (dry)", "calories": 389, "protein_g": 16.9, "carbs_g": 66.3, "fat_g": 6.9, "fiber_g": 10.6, "category": "grain"},
    {"name": "Oatmeal (cooked)", "calories": 71, "protein_g": 2.5, "carbs_g": 12.0, "fat_g": 1.5, "fiber_g": 1.7, "category": "grain"},
    {"name": "Whole wheat bread", "calories": 247, "protein_g": 13.0, "carbs_g": 41.0, "fat_g": 4.2, "fiber_g": 6.0, "category": "grain"},
    {"name": "White bread", "calories": 265, "protein_g": 9.0, "carbs_g": 49.0, "fat_g": 3.2, "fiber_g": 2.7, "category": "grain"},
    {"name": "Pasta (cooked)", "calories": 131, "protein_g": 5.0, "carbs_g": 25.0, "fat_g": 1.1, "fiber_g": 1.8, "category": "grain"},
    {"name": "Whole wheat pasta (cooked)", "calories": 124, "protein_g": 5.3, "carbs_g": 26.5, "fat_g": 0.5, "fiber_g": 3.9, "category": "grain"},
    {"name": "Quinoa (cooked)", "calories": 120, "protein_g": 4.4, "carbs_g": 21.3, "fat_g": 1.9, "fiber_g": 2.8, "category": "grain"},
    {"name": "Cornmeal", "calories": 362, "protein_g": 8.1, "carbs_g": 76.9, "fat_g": 3.6, "fiber_g": 7.3, "category": "grain"},
    {"name": "Barley (cooked)", "calories": 123, "protein_g": 2.3, "carbs_g": 28.2, "fat_g": 0.4, "fiber_g": 3.8, "category": "grain"},
    {"name": "Tortilla (flour)", "calories": 312, "protein_g": 8.0, "carbs_g": 51.0, "fat_g": 8.3, "fiber_g": 2.3, "category": "grain"},
    {"name": "Bagel", "calories": 250, "protein_g": 10.0, "carbs_g": 48.0, "fat_g": 1.6, "fiber_g": 2.1, "category": "grain"},
    {"name": "Couscous (cooked)", "calories": 112, "protein_g": 3.8, "carbs_g": 23.2, "fat_g": 0.2, "fiber_g": 1.4, "category": "grain"},
    {"name": "Millet (cooked)", "calories": 119, "protein_g": 3.5, "carbs_g": 23.7, "fat_g": 1.0, "fiber_g": 1.3, "category": "grain"},
    {"name": "Bulgur (cooked)", "calories": 83, "protein_g": 3.1, "carbs_g": 18.6, "fat_g": 0.2, "fiber_g": 4.5, "category": "grain"},
    {"name": "Crackers (whole grain)", "calories": 413, "protein_g": 10.0, "carbs_g": 68.0, "fat_g": 10.0, "fiber_g": 5.0, "category": "grain"},

    # ── Proteins — meat & poultry ─────────────────────────────────────────────
    {"name": "Chicken breast (cooked)", "calories": 165, "protein_g": 31.0, "carbs_g": 0.0, "fat_g": 3.6, "fiber_g": 0.0, "category": "protein"},
    {"name": "Chicken thigh (cooked)", "calories": 209, "protein_g": 26.0, "carbs_g": 0.0, "fat_g": 11.0, "fiber_g": 0.0, "category": "protein"},
    {"name": "Ground beef 80/20 (cooked)", "calories": 254, "protein_g": 26.0, "carbs_g": 0.0, "fat_g": 17.0, "fiber_g": 0.0, "category": "protein"},
    {"name": "Ground turkey (cooked)", "calories": 189, "protein_g": 27.0, "carbs_g": 0.0, "fat_g": 9.3, "fiber_g": 0.0, "category": "protein"},
    {"name": "Steak (sirloin, cooked)", "calories": 207, "protein_g": 30.0, "carbs_g": 0.0, "fat_g": 9.3, "fiber_g": 0.0, "category": "protein"},
    {"name": "Pork tenderloin (cooked)", "calories": 143, "protein_g": 26.0, "carbs_g": 0.0, "fat_g": 3.5, "fiber_g": 0.0, "category": "protein"},
    {"name": "Bacon (cooked)", "calories": 541, "protein_g": 37.0, "carbs_g": 1.4, "fat_g": 42.0, "fiber_g": 0.0, "category": "protein"},
    {"name": "Ham (sliced)", "calories": 145, "protein_g": 21.0, "carbs_g": 1.5, "fat_g": 6.0, "fiber_g": 0.0, "category": "protein"},
    {"name": "Lamb (leg, cooked)", "calories": 217, "protein_g": 29.0, "carbs_g": 0.0, "fat_g": 11.0, "fiber_g": 0.0, "category": "protein"},

    # ── Proteins — fish & seafood ─────────────────────────────────────────────
    {"name": "Salmon (cooked)", "calories": 208, "protein_g": 28.0, "carbs_g": 0.0, "fat_g": 10.0, "fiber_g": 0.0, "category": "protein"},
    {"name": "Tuna (canned in water)", "calories": 116, "protein_g": 26.0, "carbs_g": 0.0, "fat_g": 1.0, "fiber_g": 0.0, "category": "protein"},
    {"name": "Tilapia (cooked)", "calories": 128, "protein_g": 26.2, "carbs_g": 0.0, "fat_g": 2.7, "fiber_g": 0.0, "category": "protein"},
    {"name": "Cod (cooked)", "calories": 105, "protein_g": 23.0, "carbs_g": 0.0, "fat_g": 0.9, "fiber_g": 0.0, "category": "protein"},
    {"name": "Shrimp (cooked)", "calories": 99, "protein_g": 24.0, "carbs_g": 0.2, "fat_g": 0.3, "fiber_g": 0.0, "category": "protein"},
    {"name": "Crab (cooked)", "calories": 97, "protein_g": 19.4, "carbs_g": 0.0, "fat_g": 1.5, "fiber_g": 0.0, "category": "protein"},
    {"name": "Sardines (canned)", "calories": 208, "protein_g": 24.6, "carbs_g": 0.0, "fat_g": 11.4, "fiber_g": 0.0, "category": "protein"},
    {"name": "Mackerel (cooked)", "calories": 262, "protein_g": 25.0, "carbs_g": 0.0, "fat_g": 17.8, "fiber_g": 0.0, "category": "protein"},
    {"name": "Halibut (cooked)", "calories": 140, "protein_g": 27.0, "carbs_g": 0.0, "fat_g": 2.9, "fiber_g": 0.0, "category": "protein"},

    # ── Eggs & dairy ──────────────────────────────────────────────────────────
    {"name": "Egg (whole)", "calories": 147, "protein_g": 13.0, "carbs_g": 1.1, "fat_g": 10.0, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Egg whites", "calories": 52, "protein_g": 11.0, "carbs_g": 0.7, "fat_g": 0.2, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Whole milk", "calories": 61, "protein_g": 3.2, "carbs_g": 4.8, "fat_g": 3.3, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Skim milk", "calories": 34, "protein_g": 3.4, "carbs_g": 5.0, "fat_g": 0.1, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Greek yogurt (plain, full fat)", "calories": 97, "protein_g": 9.0, "carbs_g": 3.6, "fat_g": 5.0, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Greek yogurt (plain, non-fat)", "calories": 59, "protein_g": 10.2, "carbs_g": 3.6, "fat_g": 0.4, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Cheddar cheese", "calories": 403, "protein_g": 25.0, "carbs_g": 1.3, "fat_g": 33.0, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Mozzarella cheese", "calories": 280, "protein_g": 28.0, "carbs_g": 2.2, "fat_g": 17.0, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Cottage cheese", "calories": 98, "protein_g": 11.1, "carbs_g": 3.4, "fat_g": 4.3, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Butter", "calories": 717, "protein_g": 0.9, "carbs_g": 0.1, "fat_g": 81.0, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Cream cheese", "calories": 342, "protein_g": 6.2, "carbs_g": 4.1, "fat_g": 34.0, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Parmesan cheese", "calories": 431, "protein_g": 38.0, "carbs_g": 4.1, "fat_g": 29.0, "fiber_g": 0.0, "category": "dairy"},

    # ── Legumes ───────────────────────────────────────────────────────────────
    {"name": "Black beans (cooked)", "calories": 132, "protein_g": 8.9, "carbs_g": 24.0, "fat_g": 0.5, "fiber_g": 8.7, "category": "legume"},
    {"name": "Chickpeas (cooked)", "calories": 164, "protein_g": 8.9, "carbs_g": 27.4, "fat_g": 2.6, "fiber_g": 7.6, "category": "legume"},
    {"name": "Lentils (cooked)", "calories": 116, "protein_g": 9.0, "carbs_g": 20.0, "fat_g": 0.4, "fiber_g": 7.9, "category": "legume"},
    {"name": "Kidney beans (cooked)", "calories": 127, "protein_g": 8.7, "carbs_g": 22.8, "fat_g": 0.5, "fiber_g": 6.4, "category": "legume"},
    {"name": "Edamame (cooked)", "calories": 121, "protein_g": 11.9, "carbs_g": 8.9, "fat_g": 5.2, "fiber_g": 5.2, "category": "legume"},
    {"name": "Tofu (firm)", "calories": 76, "protein_g": 8.2, "carbs_g": 1.9, "fat_g": 4.2, "fiber_g": 0.3, "category": "legume"},
    {"name": "Tempeh", "calories": 193, "protein_g": 20.3, "carbs_g": 7.6, "fat_g": 11.0, "fiber_g": 0.0, "category": "legume"},
    {"name": "Peanut butter", "calories": 588, "protein_g": 25.0, "carbs_g": 20.0, "fat_g": 50.0, "fiber_g": 6.0, "category": "legume"},
    {"name": "Hummus", "calories": 177, "protein_g": 7.9, "carbs_g": 14.3, "fat_g": 9.6, "fiber_g": 6.0, "category": "legume"},
    {"name": "Navy beans (cooked)", "calories": 140, "protein_g": 8.2, "carbs_g": 26.0, "fat_g": 0.6, "fiber_g": 10.5, "category": "legume"},
    {"name": "Pinto beans (cooked)", "calories": 143, "protein_g": 9.0, "carbs_g": 26.2, "fat_g": 0.6, "fiber_g": 9.0, "category": "legume"},

    # ── Nuts & seeds ──────────────────────────────────────────────────────────
    {"name": "Almonds", "calories": 579, "protein_g": 21.2, "carbs_g": 21.6, "fat_g": 49.9, "fiber_g": 12.5, "category": "nut"},
    {"name": "Walnuts", "calories": 654, "protein_g": 15.2, "carbs_g": 13.7, "fat_g": 65.2, "fiber_g": 6.7, "category": "nut"},
    {"name": "Cashews", "calories": 553, "protein_g": 18.2, "carbs_g": 30.2, "fat_g": 43.9, "fiber_g": 3.3, "category": "nut"},
    {"name": "Pistachios", "calories": 560, "protein_g": 20.6, "carbs_g": 27.2, "fat_g": 45.3, "fiber_g": 10.3, "category": "nut"},
    {"name": "Pecans", "calories": 691, "protein_g": 9.2, "carbs_g": 13.9, "fat_g": 72.0, "fiber_g": 9.6, "category": "nut"},
    {"name": "Macadamia nuts", "calories": 718, "protein_g": 7.9, "carbs_g": 13.8, "fat_g": 75.8, "fiber_g": 8.6, "category": "nut"},
    {"name": "Brazil nuts", "calories": 659, "protein_g": 14.3, "carbs_g": 11.7, "fat_g": 67.1, "fiber_g": 7.5, "category": "nut"},
    {"name": "Chia seeds", "calories": 486, "protein_g": 16.5, "carbs_g": 42.1, "fat_g": 30.7, "fiber_g": 34.4, "category": "nut"},
    {"name": "Flaxseeds", "calories": 534, "protein_g": 18.3, "carbs_g": 28.9, "fat_g": 42.2, "fiber_g": 27.3, "category": "nut"},
    {"name": "Sunflower seeds", "calories": 584, "protein_g": 20.8, "carbs_g": 20.0, "fat_g": 51.5, "fiber_g": 8.6, "category": "nut"},
    {"name": "Pumpkin seeds", "calories": 559, "protein_g": 30.2, "carbs_g": 10.7, "fat_g": 49.1, "fiber_g": 6.0, "category": "nut"},
    {"name": "Sesame seeds", "calories": 573, "protein_g": 17.7, "carbs_g": 23.5, "fat_g": 49.7, "fiber_g": 11.8, "category": "nut"},

    # ── Oils & fats ───────────────────────────────────────────────────────────
    {"name": "Olive oil", "calories": 884, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 100.0, "fiber_g": 0.0, "category": "fat"},
    {"name": "Coconut oil", "calories": 892, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 100.0, "fiber_g": 0.0, "category": "fat"},
    {"name": "Avocado oil", "calories": 884, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 100.0, "fiber_g": 0.0, "category": "fat"},
    {"name": "Sunflower oil", "calories": 884, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 100.0, "fiber_g": 0.0, "category": "fat"},

    # ── Snacks & processed foods ──────────────────────────────────────────────
    {"name": "Potato chips", "calories": 536, "protein_g": 7.0, "carbs_g": 53.0, "fat_g": 35.0, "fiber_g": 4.4, "category": "snack"},
    {"name": "Popcorn (air-popped)", "calories": 387, "protein_g": 13.0, "carbs_g": 78.1, "fat_g": 4.5, "fiber_g": 14.5, "category": "snack"},
    {"name": "Dark chocolate (70%+)", "calories": 598, "protein_g": 7.8, "carbs_g": 46.0, "fat_g": 43.0, "fiber_g": 10.9, "category": "snack"},
    {"name": "Milk chocolate", "calories": 535, "protein_g": 7.7, "carbs_g": 59.2, "fat_g": 29.7, "fiber_g": 3.4, "category": "snack"},
    {"name": "Granola bar (plain)", "calories": 471, "protein_g": 10.0, "carbs_g": 64.0, "fat_g": 20.0, "fiber_g": 4.3, "category": "snack"},
    {"name": "Protein bar", "calories": 392, "protein_g": 30.0, "carbs_g": 45.0, "fat_g": 10.0, "fiber_g": 5.0, "category": "snack"},
    {"name": "Rice cakes", "calories": 387, "protein_g": 8.0, "carbs_g": 82.0, "fat_g": 3.0, "fiber_g": 4.0, "category": "snack"},
    {"name": "Pretzels", "calories": 380, "protein_g": 9.0, "carbs_g": 80.0, "fat_g": 3.5, "fiber_g": 2.7, "category": "snack"},
    {"name": "Tortilla chips", "calories": 489, "protein_g": 7.0, "carbs_g": 65.0, "fat_g": 23.0, "fiber_g": 4.5, "category": "snack"},

    # ── Beverages ─────────────────────────────────────────────────────────────
    {"name": "Orange juice", "calories": 45, "protein_g": 0.7, "carbs_g": 10.4, "fat_g": 0.2, "fiber_g": 0.2, "category": "beverage"},
    {"name": "Apple juice", "calories": 46, "protein_g": 0.1, "carbs_g": 11.4, "fat_g": 0.1, "fiber_g": 0.2, "category": "beverage"},
    {"name": "Almond milk (unsweetened)", "calories": 15, "protein_g": 0.6, "carbs_g": 0.6, "fat_g": 1.2, "fiber_g": 0.2, "category": "beverage"},
    {"name": "Oat milk", "calories": 47, "protein_g": 1.0, "carbs_g": 7.5, "fat_g": 1.5, "fiber_g": 0.8, "category": "beverage"},
    {"name": "Soy milk", "calories": 33, "protein_g": 2.9, "carbs_g": 1.8, "fat_g": 1.9, "fiber_g": 0.4, "category": "beverage"},
    {"name": "Coffee (black)", "calories": 2, "protein_g": 0.3, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0, "category": "beverage"},
    {"name": "Green tea", "calories": 1, "protein_g": 0.2, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0, "category": "beverage"},
    {"name": "Red wine", "calories": 85, "protein_g": 0.1, "carbs_g": 2.7, "fat_g": 0.0, "fiber_g": 0.0, "category": "beverage"},
    {"name": "Beer (regular)", "calories": 43, "protein_g": 0.5, "carbs_g": 3.6, "fat_g": 0.0, "fiber_g": 0.0, "category": "beverage"},
    {"name": "Coconut water", "calories": 19, "protein_g": 0.7, "carbs_g": 3.7, "fat_g": 0.2, "fiber_g": 1.1, "category": "beverage"},

    # ── Condiments & sauces ───────────────────────────────────────────────────
    {"name": "Ketchup", "calories": 101, "protein_g": 1.7, "carbs_g": 25.9, "fat_g": 0.1, "fiber_g": 0.3, "category": "condiment"},
    {"name": "Mayonnaise", "calories": 680, "protein_g": 1.0, "carbs_g": 0.6, "fat_g": 75.0, "fiber_g": 0.0, "category": "condiment"},
    {"name": "Soy sauce", "calories": 60, "protein_g": 10.5, "carbs_g": 5.6, "fat_g": 0.1, "fiber_g": 0.8, "category": "condiment"},
    {"name": "Hot sauce", "calories": 21, "protein_g": 0.9, "carbs_g": 4.8, "fat_g": 0.1, "fiber_g": 0.4, "category": "condiment"},
    {"name": "Ranch dressing", "calories": 483, "protein_g": 1.4, "carbs_g": 4.4, "fat_g": 52.0, "fiber_g": 0.2, "category": "condiment"},
    {"name": "Salsa", "calories": 36, "protein_g": 1.6, "carbs_g": 7.3, "fat_g": 0.2, "fiber_g": 1.3, "category": "condiment"},

    # ── Baked goods & sweets ──────────────────────────────────────────────────
    {"name": "Pancakes (plain)", "calories": 227, "protein_g": 6.0, "carbs_g": 40.0, "fat_g": 5.8, "fiber_g": 1.5, "category": "baked"},
    {"name": "Waffles", "calories": 291, "protein_g": 8.0, "carbs_g": 33.0, "fat_g": 14.0, "fiber_g": 1.3, "category": "baked"},
    {"name": "Muffin (blueberry)", "calories": 377, "protein_g": 5.7, "carbs_g": 55.0, "fat_g": 15.0, "fiber_g": 1.0, "category": "baked"},
    {"name": "Croissant", "calories": 406, "protein_g": 8.2, "carbs_g": 45.0, "fat_g": 21.0, "fiber_g": 2.0, "category": "baked"},
    {"name": "Cookie (chocolate chip)", "calories": 488, "protein_g": 5.3, "carbs_g": 63.0, "fat_g": 24.0, "fiber_g": 2.0, "category": "baked"},
    {"name": "Cake (chocolate)", "calories": 371, "protein_g": 5.0, "carbs_g": 55.0, "fat_g": 15.0, "fiber_g": 2.0, "category": "baked"},
    {"name": "Donut (glazed)", "calories": 452, "protein_g": 5.2, "carbs_g": 51.0, "fat_g": 25.0, "fiber_g": 1.3, "category": "baked"},
    {"name": "Ice cream (vanilla)", "calories": 207, "protein_g": 3.5, "carbs_g": 24.0, "fat_g": 11.0, "fiber_g": 0.0, "category": "baked"},

    # ── Fast food & prepared meals ─────────────────────────────────────────────
    {"name": "Hamburger (plain)", "calories": 250, "protein_g": 12.0, "carbs_g": 25.0, "fat_g": 11.0, "fiber_g": 1.2, "category": "prepared"},
    {"name": "French fries", "calories": 312, "protein_g": 3.4, "carbs_g": 41.0, "fat_g": 15.0, "fiber_g": 3.8, "category": "prepared"},
    {"name": "Pizza (cheese)", "calories": 266, "protein_g": 11.0, "carbs_g": 33.0, "fat_g": 10.0, "fiber_g": 2.3, "category": "prepared"},
    {"name": "Burrito (bean and cheese)", "calories": 189, "protein_g": 8.3, "carbs_g": 27.0, "fat_g": 5.6, "fiber_g": 3.3, "category": "prepared"},
    {"name": "Caesar salad (with dressing)", "calories": 190, "protein_g": 6.0, "carbs_g": 8.5, "fat_g": 15.0, "fiber_g": 1.9, "category": "prepared"},
    {"name": "Chicken soup", "calories": 40, "protein_g": 3.6, "carbs_g": 4.4, "fat_g": 1.0, "fiber_g": 0.2, "category": "prepared"},
    {"name": "Mac and cheese", "calories": 164, "protein_g": 7.0, "carbs_g": 21.4, "fat_g": 5.7, "fiber_g": 0.8, "category": "prepared"},
    {"name": "Fried rice", "calories": 163, "protein_g": 4.7, "carbs_g": 27.0, "fat_g": 4.0, "fiber_g": 1.2, "category": "prepared"},
    {"name": "Sushi roll (California)", "calories": 93, "protein_g": 4.0, "carbs_g": 18.4, "fat_g": 0.6, "fiber_g": 1.2, "category": "prepared"},
    {"name": "Pad thai", "calories": 156, "protein_g": 8.7, "carbs_g": 20.0, "fat_g": 4.6, "fiber_g": 1.5, "category": "prepared"},

    # ── Breakfast-specific ────────────────────────────────────────────────────
    {"name": "Cereal (corn flakes)", "calories": 357, "protein_g": 7.5, "carbs_g": 84.0, "fat_g": 0.4, "fiber_g": 2.0, "category": "grain"},
    {"name": "Cereal (granola)", "calories": 471, "protein_g": 10.0, "carbs_g": 64.0, "fat_g": 20.0, "fiber_g": 4.3, "category": "grain"},
    {"name": "Yogurt (fruit, regular)", "calories": 99, "protein_g": 3.5, "carbs_g": 19.0, "fat_g": 1.2, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Sausage (pork)", "calories": 301, "protein_g": 15.0, "carbs_g": 2.0, "fat_g": 26.0, "fiber_g": 0.0, "category": "protein"},
    {"name": "Smoked salmon", "calories": 179, "protein_g": 25.4, "carbs_g": 0.0, "fat_g": 8.8, "fiber_g": 0.0, "category": "protein"},

    # ── Additional common foods ────────────────────────────────────────────────
    {"name": "Peanuts", "calories": 567, "protein_g": 25.8, "carbs_g": 16.1, "fat_g": 49.2, "fiber_g": 8.5, "category": "nut"},
    {"name": "Mixed nuts", "calories": 607, "protein_g": 14.1, "carbs_g": 26.4, "fat_g": 52.5, "fiber_g": 5.2, "category": "nut"},
    {"name": "Almond butter", "calories": 614, "protein_g": 20.9, "carbs_g": 21.4, "fat_g": 55.5, "fiber_g": 10.3, "category": "nut"},
    {"name": "Tahini", "calories": 595, "protein_g": 17.0, "carbs_g": 21.2, "fat_g": 53.8, "fiber_g": 9.3, "category": "nut"},
    {"name": "Guacamole", "calories": 155, "protein_g": 2.0, "carbs_g": 8.5, "fat_g": 14.1, "fiber_g": 6.0, "category": "vegetable"},
    {"name": "Salad (garden, no dressing)", "calories": 20, "protein_g": 1.5, "carbs_g": 3.0, "fat_g": 0.3, "fiber_g": 1.8, "category": "vegetable"},
    {"name": "Smoothie (fruit, no dairy)", "calories": 60, "protein_g": 0.8, "carbs_g": 14.0, "fat_g": 0.4, "fiber_g": 1.5, "category": "beverage"},
    {"name": "Protein shake (whey)", "calories": 120, "protein_g": 24.0, "carbs_g": 5.0, "fat_g": 2.0, "fiber_g": 0.5, "category": "dairy"},
    {"name": "Falafel", "calories": 333, "protein_g": 13.3, "carbs_g": 31.8, "fat_g": 17.8, "fiber_g": 4.9, "category": "legume"},
    {"name": "Gyro wrap", "calories": 220, "protein_g": 13.0, "carbs_g": 25.0, "fat_g": 7.5, "fiber_g": 2.0, "category": "prepared"},
    {"name": "Spring roll (fried)", "calories": 227, "protein_g": 5.0, "carbs_g": 24.0, "fat_g": 12.5, "fiber_g": 1.5, "category": "prepared"},
    {"name": "Dim sum (har gow)", "calories": 135, "protein_g": 7.0, "carbs_g": 17.0, "fat_g": 4.0, "fiber_g": 0.8, "category": "prepared"},
    {"name": "Naan bread", "calories": 317, "protein_g": 9.5, "carbs_g": 50.0, "fat_g": 9.0, "fiber_g": 2.0, "category": "grain"},
    {"name": "Pita bread", "calories": 275, "protein_g": 9.1, "carbs_g": 55.7, "fat_g": 1.2, "fiber_g": 2.2, "category": "grain"},
    {"name": "Sourdough bread", "calories": 269, "protein_g": 10.0, "carbs_g": 51.0, "fat_g": 2.0, "fiber_g": 2.4, "category": "grain"},
    {"name": "Energy bar (Larabar-type)", "calories": 410, "protein_g": 6.0, "carbs_g": 63.0, "fat_g": 15.0, "fiber_g": 8.0, "category": "snack"},
    {"name": "Gummy bears", "calories": 350, "protein_g": 6.5, "carbs_g": 77.0, "fat_g": 0.5, "fiber_g": 0.0, "category": "snack"},
    {"name": "Dried mango", "calories": 319, "protein_g": 2.7, "carbs_g": 78.6, "fat_g": 1.2, "fiber_g": 2.4, "category": "snack"},
    {"name": "Dried apricots", "calories": 241, "protein_g": 3.4, "carbs_g": 62.6, "fat_g": 0.5, "fiber_g": 7.3, "category": "snack"},
    {"name": "Raisins", "calories": 299, "protein_g": 3.1, "carbs_g": 79.2, "fat_g": 0.5, "fiber_g": 3.7, "category": "snack"},
    {"name": "Soup (vegetable)", "calories": 52, "protein_g": 2.4, "carbs_g": 9.8, "fat_g": 0.8, "fiber_g": 1.6, "category": "prepared"},
    {"name": "Minestrone soup", "calories": 62, "protein_g": 2.8, "carbs_g": 10.9, "fat_g": 1.2, "fiber_g": 1.9, "category": "prepared"},
    {"name": "Stir fry (vegetable, soy sauce)", "calories": 80, "protein_g": 3.0, "carbs_g": 10.0, "fat_g": 3.0, "fiber_g": 2.5, "category": "prepared"},
    {"name": "Curry (chicken, mild)", "calories": 160, "protein_g": 13.0, "carbs_g": 10.0, "fat_g": 7.0, "fiber_g": 2.0, "category": "prepared"},
    {"name": "Chili (beef and beans)", "calories": 113, "protein_g": 7.5, "carbs_g": 11.3, "fat_g": 4.0, "fiber_g": 3.2, "category": "prepared"},
    {"name": "Risotto", "calories": 149, "protein_g": 4.0, "carbs_g": 26.0, "fat_g": 3.5, "fiber_g": 0.6, "category": "prepared"},
    {"name": "Scrambled eggs", "calories": 149, "protein_g": 10.1, "carbs_g": 1.6, "fat_g": 11.5, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Omelet (plain)", "calories": 154, "protein_g": 11.4, "carbs_g": 0.5, "fat_g": 12.0, "fiber_g": 0.0, "category": "dairy"},
    {"name": "Toast with butter", "calories": 315, "protein_g": 8.0, "carbs_g": 40.0, "fat_g": 14.0, "fiber_g": 2.5, "category": "grain"},
    {"name": "Avocado toast", "calories": 190, "protein_g": 5.0, "carbs_g": 22.0, "fat_g": 9.5, "fiber_g": 5.5, "category": "grain"},
    {"name": "Kimchi", "calories": 15, "protein_g": 1.1, "carbs_g": 2.4, "fat_g": 0.5, "fiber_g": 1.6, "category": "vegetable"},
    {"name": "Miso soup", "calories": 40, "protein_g": 3.0, "carbs_g": 5.6, "fat_g": 1.1, "fiber_g": 0.8, "category": "prepared"},
    {"name": "Tofu scramble", "calories": 84, "protein_g": 9.0, "carbs_g": 3.0, "fat_g": 4.5, "fiber_g": 0.7, "category": "legume"},
]

# Build lookup index (lower-case name → entry)
FOODS: Dict[str, Dict] = {entry["name"].lower(): entry for entry in _FOODS_RAW}

# Sorted list of all food names for fuzzy matching
_ALL_NAMES: List[str] = sorted(FOODS.keys())


# ── Portion-to-gram table ─────────────────────────────────────────────────────
# Generic portion sizes that work for most foods.
# These are deliberately conservative estimates.
PORTION_GRAMS: Dict[str, float] = {
    # Size qualifiers
    "small":          80.0,
    "medium":        120.0,
    "large":         180.0,
    "extra large":   250.0,
    # Household measures
    "cup":           240.0,
    "half cup":      120.0,
    "tablespoon":     15.0,
    "teaspoon":        5.0,
    "ounce":          28.3,
    "oz":             28.3,
    "gram":            1.0,
    "g":               1.0,
    "kg":           1000.0,
    # Counts (very rough — food-independent fallback)
    "piece":          80.0,
    "slice":          30.0,
    "handful":        30.0,
    "bite":           10.0,
}


# ── Public API ────────────────────────────────────────────────────────────────

def lookup(food_name: str, cutoff: float = 0.55) -> Optional[Dict]:
    """Return the nutritional entry for *food_name* (per 100 g).

    Performs case-insensitive exact match first; falls back to fuzzy match
    with difflib.  Returns ``None`` when confidence is below *cutoff*.

    The returned dict includes: name, calories, protein_g, carbs_g, fat_g,
    fiber_g, category.  These are PER 100 g values.
    """
    key = food_name.strip().lower()
    if key in FOODS:
        return dict(FOODS[key])

    # Fuzzy match
    matches = difflib.get_close_matches(key, _ALL_NAMES, n=1, cutoff=cutoff)
    if matches:
        return dict(FOODS[matches[0]])

    # Try prefix / substring match
    for name in _ALL_NAMES:
        if key in name or name in key:
            return dict(FOODS[name])

    return None


def lookup_with_portion(food_name: str, grams: float) -> Optional[Dict]:
    """Return nutritional values scaled to *grams*.

    Returns the same dict structure as ``lookup()`` but with all
    macro/calorie values scaled to the requested portion size.
    Adds a ``portion_g`` field.  Returns ``None`` when the food is not
    found.
    """
    entry = lookup(food_name)
    if entry is None:
        return None
    scale = grams / 100.0
    return {
        "name": entry["name"],
        "portion_g": grams,
        "calories": round(entry["calories"] * scale, 1),
        "protein_g": round(entry["protein_g"] * scale, 1),
        "carbs_g": round(entry["carbs_g"] * scale, 1),
        "fat_g": round(entry["fat_g"] * scale, 1),
        "fiber_g": round(entry["fiber_g"] * scale, 1),
        "category": entry["category"],
    }


def portion_size_to_grams(portion_label: str, food_name: str = "") -> float:
    """Convert a human-readable portion label to grams.

    Examples:
        "medium"       → 120
        "1 cup"        → 240
        "2 tablespoons" → 30
        "150g"         → 150

    Returns 100.0 (one standard serving) when the label is unrecognised.
    """
    label = portion_label.strip().lower()

    # Handle "Ng" / "N g" / "N grams"
    import re
    gram_match = re.match(r"^(\d+\.?\d*)\s*g(?:rams?)?$", label)
    if gram_match:
        return float(gram_match.group(1))

    # Handle "N cups" / "N tablespoons" / "N teaspoons" etc.
    multi_match = re.match(r"^(\d+\.?\d*)\s+(.+)$", label)
    if multi_match:
        count = float(multi_match.group(1))
        unit = multi_match.group(2).rstrip("s")  # remove plural
        g_per_unit = PORTION_GRAMS.get(unit, PORTION_GRAMS.get(unit + "s", None))
        if g_per_unit is not None:
            return count * g_per_unit

    # Direct lookup (singular and plural)
    for key in [label, label.rstrip("s")]:
        if key in PORTION_GRAMS:
            return PORTION_GRAMS[key]

    return 100.0  # default: 100 g = 1 standard serving


def list_by_category(category: str) -> List[Dict]:
    """Return all foods in a given category."""
    return [dict(v) for v in FOODS.values() if v["category"] == category]


def all_food_names() -> List[str]:
    """Return sorted list of all food names (original casing)."""
    return sorted(entry["name"] for entry in _FOODS_RAW)
