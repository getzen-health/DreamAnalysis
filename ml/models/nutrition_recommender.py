"""Evidence-based nutrition-mood recommendation engine.

Maps current emotional state (valence, arousal, stress, focus) to specific
food and nutrient recommendations backed by clinical trial evidence.

Based on:
- SMILES trial (Jacka et al., 2017): Mediterranean diet reduces depression
- Omega-3 meta-analysis (Liao et al., 2019): EPA/DHA d=0.28 for depression
- Tryptophan RCTs (Kikuchi et al., 2021): serotonin precursor effect
- Magnesium RCTs (Tarleton et al., 2017): anxiety reduction in 6 weeks
- Polyphenols (Pase et al., 2015): flavonoids improve mood and cognition
- Dietary patterns review (Jacka et al., 2022): ultra-processed food → depression risk

All recommendations include:
- Evidence level: meta-analysis / RCT / observational
- Mechanism: biological pathway
- Effect size where known
- Foods that contain the nutrient
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ── Knowledge base ────────────────────────────────────────────────────────────
# Each entry: id, nutrient, mechanism, evidence_level, effect_size,
#             mood_dimensions (which emotional dimensions it addresses),
#             foods (list of food sources), timing, caution

_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "omega3",
        "nutrient": "Omega-3 (EPA/DHA)",
        "mechanism": "Anti-inflammatory; supports serotonin and dopamine receptor function",
        "evidence_level": "meta-analysis",
        "effect_size": "d=0.28 (depression), d=0.31 (anxiety)",
        "mood_dimensions": ["low_valence", "high_stress", "high_arousal_neg"],
        "foods": ["salmon", "mackerel", "sardines", "walnuts", "flaxseed", "chia seeds"],
        "timing": "With a meal (fat-soluble absorption)",
        "caution": "High doses >3g/day may affect bleeding — consult doctor if on blood thinners",
        "priority_score": 0.90,
    },
    {
        "id": "tryptophan",
        "nutrient": "Tryptophan-rich foods",
        "mechanism": "Serotonin precursor — increases 5-HT synthesis; requires B6 cofactor",
        "evidence_level": "RCT",
        "effect_size": "Mood improvement in 3-4 hours post-consumption (Kikuchi 2021)",
        "mood_dimensions": ["low_valence", "high_stress"],
        "foods": ["turkey", "chicken", "eggs", "milk", "cheese", "banana", "oats", "tofu"],
        "timing": "Combine with low-glycemic carbs (enhances brain uptake vs competing AAs)",
        "caution": "Most effective when not competing with high-protein meal",
        "priority_score": 0.85,
    },
    {
        "id": "magnesium",
        "nutrient": "Magnesium",
        "mechanism": "NMDA receptor modulator; reduces HPA axis hyperactivity (cortisol)",
        "evidence_level": "RCT",
        "effect_size": "Anxiety reduction in 6 weeks at 300mg/day (Tarleton 2017)",
        "mood_dimensions": ["high_stress", "high_arousal_neg", "low_valence"],
        "foods": ["dark chocolate", "spinach", "almonds", "cashews", "pumpkin seeds", "black beans", "avocado"],
        "timing": "Evening (supports sleep quality too)",
        "caution": "High doses (>400mg) may cause loose stools",
        "priority_score": 0.88,
    },
    {
        "id": "b_vitamins",
        "nutrient": "B vitamins (B6, B9 folate, B12)",
        "mechanism": "Cofactors for monoamine synthesis (serotonin, dopamine, norepinephrine)",
        "evidence_level": "meta-analysis",
        "effect_size": "Folate deficiency doubles depression risk (Papakostas 2012)",
        "mood_dimensions": ["low_valence", "low_focus", "fatigue"],
        "foods": ["leafy greens", "lentils", "eggs", "salmon", "beef liver", "chickpeas", "nutritional yeast"],
        "timing": "Morning (energy metabolism support)",
        "caution": "B12 especially important for plant-based diets",
        "priority_score": 0.82,
    },
    {
        "id": "polyphenols",
        "nutrient": "Flavonoids / Polyphenols",
        "mechanism": "BDNF upregulation, reduced neuroinflammation, improved cerebral blood flow",
        "evidence_level": "RCT",
        "effect_size": "Flavonoids improve mood scores in 4 weeks (Pase 2015)",
        "mood_dimensions": ["low_valence", "low_focus", "high_stress"],
        "foods": ["blueberries", "dark chocolate (70%+)", "green tea", "red wine (1 glass)", "purple grapes", "citrus", "apples"],
        "timing": "Any time — berries best pre-task for acute cognition",
        "caution": "Red wine benefits only at low consumption (≤1/day)",
        "priority_score": 0.80,
    },
    {
        "id": "fermented_foods",
        "nutrient": "Probiotic / Fermented foods",
        "mechanism": "Gut-brain axis: microbiome diversity → 95% of serotonin made in gut",
        "evidence_level": "RCT",
        "effect_size": "Probiotic supplementation reduces depression scores (Liang 2019)",
        "mood_dimensions": ["low_valence", "high_stress", "high_arousal_neg"],
        "foods": ["yogurt (live cultures)", "kefir", "kimchi", "sauerkraut", "miso", "tempeh", "kombucha"],
        "timing": "Daily consistency matters more than timing",
        "caution": "Start slow if gut-sensitive — some fermented foods cause initial bloating",
        "priority_score": 0.78,
    },
    {
        "id": "complex_carbs",
        "nutrient": "Complex carbohydrates",
        "mechanism": "Steady glucose → stable mood; triggers insulin → brain tryptophan uptake",
        "evidence_level": "observational",
        "effect_size": "Low GI diet reduces mood variability (Brand-Miller 2011)",
        "mood_dimensions": ["high_arousal_neg", "high_stress", "mood_instability"],
        "foods": ["oats", "brown rice", "quinoa", "sweet potato", "lentils", "whole grain bread"],
        "timing": "Avoid simple carbs (sugar spikes) when stressed",
        "caution": "Glycemic index matters — refined carbs can worsen mood swings",
        "priority_score": 0.72,
    },
    {
        "id": "vitamin_d",
        "nutrient": "Vitamin D",
        "mechanism": "VDR receptors in limbic system; modulates serotonin synthesis gene expression",
        "evidence_level": "meta-analysis",
        "effect_size": "Deficiency associated with 1.5× depression risk (Shaffer 2014)",
        "mood_dimensions": ["low_valence", "low_focus", "fatigue"],
        "foods": ["salmon", "tuna", "eggs", "fortified milk", "mushrooms (UV-exposed)", "fortified cereals"],
        "timing": "Morning with fat (fat-soluble)",
        "caution": "Sunlight is primary source — 15min outdoor exposure also recommended",
        "priority_score": 0.76,
    },
    {
        "id": "hydration",
        "nutrient": "Water / Hydration",
        "mechanism": "Even mild dehydration (1-2%) impairs cognitive performance and mood",
        "evidence_level": "RCT",
        "effect_size": "Rehydration improves alertness and reduces fatigue in <20 min (Popkin 2010)",
        "mood_dimensions": ["low_focus", "fatigue", "high_stress"],
        "foods": ["water", "herbal tea", "coconut water", "cucumber", "watermelon", "broth"],
        "timing": "Throughout day; especially 30min before cognitive tasks",
        "caution": "Caffeinated drinks contribute to dehydration — don't count toward total",
        "priority_score": 0.85,
    },
    {
        "id": "l_theanine",
        "nutrient": "L-theanine (+ caffeine)",
        "mechanism": "GABA modulation + alpha brain waves → alert calm; caffeine synergy",
        "evidence_level": "RCT",
        "effect_size": "100mg L-theanine + 50mg caffeine improves focus and reduces anxiety (Giesbrecht 2010)",
        "mood_dimensions": ["high_stress", "low_focus", "high_arousal_neg"],
        "foods": ["green tea", "matcha", "white tea"],
        "timing": "60min before tasks requiring focus",
        "caution": "Avoid >3pm if caffeine-sensitive (may affect sleep)",
        "priority_score": 0.83,
    },
]


def _emotion_to_dimensions(emotion_data: Dict) -> List[str]:
    """Map emotion metrics to mood dimension labels for recommendation matching."""
    dims = []
    valence = emotion_data.get("valence", 0.0)
    arousal = emotion_data.get("arousal", 0.5)
    stress = emotion_data.get("stress_index", 0.5)
    focus = emotion_data.get("focus_index", 0.5)
    relax = emotion_data.get("relaxation_index", 0.5)

    if valence < -0.15:
        dims.append("low_valence")
    if stress > 0.55:
        dims.append("high_stress")
    if arousal > 0.65 and valence < 0:
        dims.append("high_arousal_neg")
    if focus < 0.40:
        dims.append("low_focus")
    if relax < 0.35:
        dims.append("fatigue")
    if abs(valence) > 0.5:  # extreme emotion = instability
        dims.append("mood_instability")

    # If no specific dimensions, default to general wellbeing
    if not dims:
        dims.append("low_valence")  # mild support for any state

    return dims


class NutritionRecommender:
    """Generate evidence-based food recommendations from emotional state.

    Usage::

        rec = NutritionRecommender()
        result = rec.recommend({"valence": -0.3, "stress_index": 0.7, ...})
        # result["recommendations"] — list of 3-5 foods/nutrients
    """

    def recommend(
        self,
        emotion_data: Dict,
        top_k: int = 4,
    ) -> Dict:
        """Return ranked food recommendations for the current emotional state.

        Args:
            emotion_data: Dict with valence, arousal, stress_index, focus_index,
                relaxation_index from emotion analysis.
            top_k: Number of recommendations to return (default 4).

        Returns:
            Dict with: recommendations (list), mood_dimensions (detected),
            emotion_summary (plain text), general_advice.
        """
        dims = _emotion_to_dimensions(emotion_data)

        # Score each nutrient by how many of its mood_dimensions match
        scored: List[tuple] = []
        for item in _KNOWLEDGE_BASE:
            overlap = len(set(item["mood_dimensions"]) & set(dims))
            if overlap > 0:
                score = overlap * item["priority_score"]
                scored.append((score, item))

        # Sort descending, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item for _, item in scored[:top_k]]

        # Always include hydration if not already in top results (universal benefit)
        hydration = next((i for i in _KNOWLEDGE_BASE if i["id"] == "hydration"), None)
        if hydration and hydration not in top and len(top) < top_k:
            top.append(hydration)

        recommendations = [
            {
                "nutrient": r["nutrient"],
                "foods": r["foods"][:5],  # top 5 food sources
                "mechanism": r["mechanism"],
                "evidence_level": r["evidence_level"],
                "effect_size": r["effect_size"],
                "timing": r["timing"],
                "caution": r["caution"],
            }
            for r in top
        ]

        emotion_summary = _summarize_state(emotion_data)
        general_advice = _general_advice(dims)

        return {
            "recommendations": recommendations,
            "mood_dimensions": dims,
            "emotion_summary": emotion_summary,
            "general_advice": general_advice,
            "n_recommendations": len(recommendations),
        }

    def get_knowledge_base(self) -> List[Dict]:
        """Return the full evidence-based nutrient knowledge base."""
        return [
            {k: v for k, v in item.items() if k != "priority_score"}
            for item in _KNOWLEDGE_BASE
        ]


def _summarize_state(d: Dict) -> str:
    valence = d.get("valence", 0.0)
    stress = d.get("stress_index", 0.5)
    focus = d.get("focus_index", 0.5)

    parts = []
    if valence < -0.3:
        parts.append("low mood")
    elif valence > 0.3:
        parts.append("positive mood")
    else:
        parts.append("neutral mood")

    if stress > 0.6:
        parts.append("elevated stress")
    if focus < 0.4:
        parts.append("reduced focus")

    return "Current state: " + ", ".join(parts) + "."


def _general_advice(dims: List[str]) -> str:
    if "high_stress" in dims:
        return "When stressed, avoid ultra-processed foods and excess caffeine — they amplify cortisol spikes. Focus on whole foods with magnesium and omega-3."
    if "low_focus" in dims:
        return "For focus: green tea (L-theanine + caffeine), hydration, and protein-rich breakfast. Avoid sugar spikes."
    if "low_valence" in dims:
        return "Evidence-based mood support: Mediterranean-style eating patterns show the strongest diet-mood link (SMILES trial, d=0.52 reduction in depression scores)."
    return "Maintaining stable blood glucose through regular whole-food meals supports consistent mood across the day."


# Module-level singleton
_recommender: Optional[NutritionRecommender] = None


def get_recommender() -> NutritionRecommender:
    global _recommender
    if _recommender is None:
        _recommender = NutritionRecommender()
    return _recommender
