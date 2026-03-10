"""Evidence-based supplement-mood knowledge base.

Pre-loaded correlations from clinical research (CANMAT 2022, 2024-2025 systematic reviews).
Provides immediate guidance before user accumulates personal data.
"""
from __future__ import annotations
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Evidence grades
# ---------------------------------------------------------------------------
# A = meta-analysis / multiple RCTs
# B = single RCT or multiple observational
# C = expert consensus / observational
# W = well-established in literature

# ---------------------------------------------------------------------------
# Supplement database
# ---------------------------------------------------------------------------
SUPPLEMENT_DB: Dict[str, dict] = {
    "omega-3": {
        "display_name": "Omega-3 (EPA>DHA)",
        "evidence_grade": "A",
        "evidence_summary": "Multiple meta-analyses; CANMAT 2022 Grade A",
        "expected_effects": {"valence": +0.10, "stress_index": -0.08, "focus_index": 0.0},
        "onset_days": 28,
        "peak_effect_weeks": 8,
        "mechanism": "Serotonin/dopamine modulation, anti-inflammatory (EPA reduces IL-6)",
        "eeg_signature": {"alpha_increase": True, "high_beta_decrease": True},
        "voice_signature": {"valence_shift": +0.05, "speech_rate_change": 0.0},
        "synergies": ["vitamin-d"],
        "cautions": ["blood-thinners"],
        "references": ["CANMAT 2022", "ScienceDirect 2025 Omega-3 and Mood"],
    },
    "vitamin-d": {
        "display_name": "Vitamin D (2000–4000 IU)",
        "evidence_grade": "A",
        "evidence_summary": "Meta-analyses link deficiency to depression; supplementation reduces risk",
        "expected_effects": {"valence": +0.08, "stress_index": -0.06, "focus_index": +0.03},
        "onset_days": 28,
        "peak_effect_weeks": 12,
        "mechanism": "TPH2 → serotonin synthesis; VDR expression in limbic system; neuroplasticity",
        "eeg_signature": {"alpha_increase": True, "high_beta_decrease": False},
        "voice_signature": {"valence_shift": +0.04, "speech_rate_change": 0.0},
        "synergies": ["omega-3", "magnesium"],
        "cautions": ["hypercalcemia-risk-high-dose"],
        "references": ["Frontiers Nutrition 2025 Vitamin D", "CANMAT 2022"],
    },
    "magnesium": {
        "display_name": "Magnesium (200–400 mg)",
        "evidence_grade": "B",
        "evidence_summary": "Multiple RCTs show anxiety/stress reduction; faster onset than other supplements",
        "expected_effects": {"valence": +0.04, "stress_index": -0.12, "focus_index": +0.02},
        "onset_days": 7,
        "peak_effect_weeks": 2,
        "mechanism": "GABA receptor agonism; HPA axis modulation; reduces cortisol reactivity",
        "eeg_signature": {"alpha_increase": True, "high_beta_decrease": True},
        "voice_signature": {"valence_shift": +0.03, "speech_rate_change": -0.05},
        "synergies": ["vitamin-d", "l-theanine"],
        "cautions": ["kidney-disease"],
        "references": ["CANMAT 2022", "Nutrients 2017 Magnesium RCT"],
    },
    "probiotics": {
        "display_name": "Probiotics",
        "evidence_grade": "A",
        "evidence_summary": "Gut-brain axis; adjunctive antidepressant effect confirmed in multiple trials",
        "expected_effects": {"valence": +0.06, "stress_index": -0.07, "focus_index": 0.0},
        "onset_days": 28,
        "peak_effect_weeks": 8,
        "mechanism": "Gut microbiome modulation → cortisol reduction; SCFA → brain neuroplasticity",
        "eeg_signature": {"alpha_increase": False, "high_beta_decrease": True},
        "voice_signature": {"valence_shift": +0.03, "speech_rate_change": 0.0},
        "synergies": [],
        "cautions": ["immunocompromised"],
        "references": ["Frontiers Nutrition 2025 Gut-Brain", "CANMAT 2022"],
    },
    "zinc": {
        "display_name": "Zinc (25–50 mg)",
        "evidence_grade": "A",
        "evidence_summary": "Adjunctive antidepressant effect; low zinc correlates with depression severity",
        "expected_effects": {"valence": +0.07, "stress_index": -0.04, "focus_index": +0.05},
        "onset_days": 42,
        "peak_effect_weeks": 12,
        "mechanism": "BDNF expression; NMDA receptor modulation; glutamate neurotransmission",
        "eeg_signature": {"alpha_increase": False, "high_beta_decrease": False},
        "voice_signature": {"valence_shift": +0.03, "speech_rate_change": 0.0},
        "synergies": [],
        "cautions": ["copper-depletion-long-term"],
        "references": ["CANMAT 2022", "Biological Psychiatry 2013 Zinc"],
    },
    "same": {
        "display_name": "SAMe (800–1600 mg)",
        "evidence_grade": "A",
        "evidence_summary": "Multiple RCTs; equivalent to TCAs in some studies; fast onset",
        "expected_effects": {"valence": +0.12, "stress_index": -0.05, "focus_index": +0.04},
        "onset_days": 14,
        "peak_effect_weeks": 4,
        "mechanism": "Methylation reactions → serotonin/dopamine/norepinephrine synthesis",
        "eeg_signature": {"alpha_increase": True, "high_beta_decrease": False},
        "voice_signature": {"valence_shift": +0.06, "speech_rate_change": +0.05},
        "synergies": ["b12", "folate"],
        "cautions": ["bipolar-disorder-risk", "ssri-interaction"],
        "references": ["CANMAT 2022", "Am J Psychiatry 2010 SAMe"],
    },
    "l-theanine": {
        "display_name": "L-Theanine (200 mg)",
        "evidence_grade": "B",
        "evidence_summary": "Acute anxiolytic + attention effects confirmed; often paired with caffeine",
        "expected_effects": {"valence": +0.05, "stress_index": -0.10, "focus_index": +0.08},
        "onset_days": 0,
        "peak_effect_weeks": 0,
        "mechanism": "Alpha-wave increase; GABA agonism; reduces caffeine jitteriness",
        "eeg_signature": {"alpha_increase": True, "high_beta_decrease": True},
        "voice_signature": {"valence_shift": +0.04, "speech_rate_change": -0.08},
        "synergies": ["caffeine"],
        "cautions": [],
        "references": ["CANMAT 2022", "J Hum Nutr Diet 2019 Theanine"],
    },
    "creatine": {
        "display_name": "Creatine (3–5 g)",
        "evidence_grade": "B",
        "evidence_summary": "Emerging evidence for antidepressant effect; well-established cognition benefit",
        "expected_effects": {"valence": +0.06, "stress_index": -0.03, "focus_index": +0.10},
        "onset_days": 14,
        "peak_effect_weeks": 4,
        "mechanism": "Brain phosphocreatine → ATP → energy metabolism; NMDA modulation",
        "eeg_signature": {"alpha_increase": False, "high_beta_decrease": False},
        "voice_signature": {"valence_shift": +0.03, "speech_rate_change": 0.0},
        "synergies": [],
        "cautions": ["kidney-disease"],
        "references": ["CANMAT 2022", "Biol Psychiatry 2012 Creatine"],
    },
    "caffeine": {
        "display_name": "Caffeine (100–400 mg)",
        "evidence_grade": "W",
        "evidence_summary": "Well-established; acute alertness but anxiety risk at high doses",
        "expected_effects": {"valence": +0.04, "stress_index": +0.06, "focus_index": +0.15},
        "onset_days": 0,
        "peak_effect_weeks": 0,
        "mechanism": "Adenosine receptor antagonism → dopamine/norepinephrine release",
        "eeg_signature": {"alpha_increase": False, "high_beta_decrease": False},
        "voice_signature": {"valence_shift": +0.02, "speech_rate_change": +0.10},
        "synergies": ["l-theanine"],
        "cautions": ["anxiety-disorder", "afternoon-use-disrupts-sleep"],
        "references": ["Neuropharmacology 2015 Caffeine"],
    },
    "melatonin": {
        "display_name": "Melatonin (0.5–5 mg)",
        "evidence_grade": "W",
        "evidence_summary": "Well-established sleep onset improvement; minimal mood effect beyond sleep",
        "expected_effects": {"valence": +0.03, "stress_index": -0.04, "focus_index": 0.0},
        "onset_days": 0,
        "peak_effect_weeks": 0,
        "mechanism": "Circadian rhythm reset via SCN photoreceptors",
        "eeg_signature": {"alpha_increase": False, "high_beta_decrease": True},
        "voice_signature": {"valence_shift": +0.02, "speech_rate_change": -0.05},
        "synergies": [],
        "cautions": ["afternoon-caffeine-reduces-effectiveness", "daytime-drowsiness"],
        "references": ["Cochrane Review 2022 Melatonin"],
    },
}

# ---------------------------------------------------------------------------
# Interaction rules
# ---------------------------------------------------------------------------
INTERACTION_RULES: List[dict] = [
    {
        "supplements": ["omega-3", "vitamin-d"],
        "type": "synergy",
        "message": "Omega-3 + Vitamin D is synergistic — both support serotonin synthesis.",
    },
    {
        "supplements": ["caffeine", "l-theanine"],
        "type": "synergy",
        "message": "Caffeine + L-Theanine is a well-known focus stack — L-Theanine smooths caffeine's edge.",
    },
    {
        "supplements": ["magnesium", "l-theanine"],
        "type": "synergy",
        "message": "Magnesium + L-Theanine both promote relaxation via GABA — mild synergy for stress.",
    },
    {
        "supplements": ["vitamin-d", "magnesium"],
        "type": "synergy",
        "message": "Magnesium is required for Vitamin D activation — taking both improves efficacy.",
    },
    {
        "supplements": ["caffeine", "melatonin"],
        "type": "caution",
        "message": "Caffeine after 2pm may reduce melatonin effectiveness — consider timing carefully.",
    },
    {
        "supplements": ["same", "caffeine"],
        "type": "caution",
        "message": "SAMe + Caffeine may amplify agitation in sensitive individuals.",
    },
]


def lookup(name: str) -> Optional[dict]:
    """Return knowledge base entry for a supplement name.

    Does fuzzy matching on common aliases (e.g. 'omega3' → 'omega-3').
    """
    key = name.lower().strip().replace(" ", "-").replace("_", "-")
    # Exact match
    if key in SUPPLEMENT_DB:
        return SUPPLEMENT_DB[key]
    # Alias match
    aliases = {
        "omega3": "omega-3",
        "fish-oil": "omega-3",
        "fish-oil-omega3": "omega-3",
        "vitamin-d3": "vitamin-d",
        "vit-d": "vitamin-d",
        "vitd": "vitamin-d",
        "mag": "magnesium",
        "magnesium-glycinate": "magnesium",
        "probiotic": "probiotics",
        "l-theanine": "l-theanine",
        "theanine": "l-theanine",
        "s-adenosyl-methionine": "same",
        "melatonin": "melatonin",
        "coffee": "caffeine",
        "espresso": "caffeine",
    }
    mapped = aliases.get(key)
    if mapped:
        return SUPPLEMENT_DB.get(mapped)
    # Partial match — return first entry whose key starts with the query
    for db_key, entry in SUPPLEMENT_DB.items():
        if db_key.startswith(key) or key in db_key:
            return entry
    return None


def check_interactions(supplement_names: List[str]) -> List[dict]:
    """Return interaction warnings/synergies for a set of supplements being logged together."""
    normalized = {
        n.lower().strip().replace(" ", "-").replace("_", "-")
        for n in supplement_names
    }
    results = []
    for rule in INTERACTION_RULES:
        if all(s in normalized for s in rule["supplements"]):
            results.append(rule)
    return results


def population_vs_personal(
    supplement_name: str,
    personal_effects: dict,
) -> dict:
    """Compare personal observed effects vs population-average expected effects.

    Args:
        supplement_name: supplement identifier
        personal_effects: dict with keys valence, stress_index, focus_index (observed shifts)

    Returns:
        comparison dict with diff, direction, and interpretation text
    """
    entry = lookup(supplement_name)
    if not entry:
        return {}

    expected = entry.get("expected_effects", {})
    comparison = {}
    for key, exp_val in expected.items():
        personal_val = personal_effects.get(key, None)
        if personal_val is None:
            continue
        diff = personal_val - exp_val
        if abs(diff) < 0.02:
            direction = "average"
            note = f"Your {key} response matches the clinical average."
        elif diff > 0:
            direction = "above_average"
            note = (
                f"Your {key} response ({personal_val:+.2f}) exceeds the clinical average "
                f"({exp_val:+.2f}) — this supplement is working particularly well for you."
            )
        else:
            direction = "below_average"
            note = (
                f"Your {key} response ({personal_val:+.2f}) is below the clinical average "
                f"({exp_val:+.2f}) — consider timing, dose, or synergistic supplements."
            )
        comparison[key] = {
            "personal": personal_val,
            "population_avg": exp_val,
            "diff": round(diff, 3),
            "direction": direction,
            "note": note,
        }
    return comparison
