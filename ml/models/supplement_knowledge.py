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
    "rhodiola": {
        "display_name": "Rhodiola Rosea",
        "evidence_grade": "B",
        "evidence_summary": "Multiple RCTs for burnout and stress; Mao 2015 vs sertraline showed comparable effect in mild-moderate depression",
        "expected_effects": {"valence": +0.03, "stress_index": -0.08, "focus_index": +0.05},
        "onset_days": 7,
        "peak_effect_weeks": 2,
        "mechanism": "Reversible MAO-A/B inhibition; salidroside + rosavin upregulate serotonin/norepinephrine; AMPK + Nrf2 antioxidant activation",
        "eeg_signature": {"alpha_increase": False, "high_beta_decrease": True},
        "voice_signature": {"valence_shift": +0.02, "speech_rate_change": +0.03},
        "synergies": [],
        "cautions": ["maoi-washout-14d", "avoid-late-evening-mild-stimulant", "cyp3a4-interactions"],
        "references": ["Phytomedicine 2015 Rhodiola vs Sertraline", "CANMAT 2022"],
    },
    "ashwagandha": {
        "display_name": "Ashwagandha (KSM-66)",
        "evidence_grade": "B",
        "evidence_summary": "Chandrasekhar 2012 RCT: 27.9% cortisol reduction; multiple stress/anxiety RCTs in healthy adults",
        "expected_effects": {"valence": +0.05, "stress_index": -0.10, "focus_index": +0.03},
        "onset_days": 14,
        "peak_effect_weeks": 8,
        "mechanism": "Withanolides modulate HPA axis; GABA-A receptor activation; NF-κB anti-inflammatory; thyroid hormone modulation",
        "eeg_signature": {"alpha_increase": True, "high_beta_decrease": True},
        "voice_signature": {"valence_shift": +0.03, "speech_rate_change": -0.03},
        "synergies": [],
        "cautions": ["avoid-in-pregnancy", "thyroid-medication-interaction", "rare-hepatotoxicity-high-dose", "potentiates-sedatives"],
        "references": ["Indian J Psychol Med 2012 Ashwagandha", "CANMAT 2022"],
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


_ALIASES: Dict[str, str] = {
    "omega3": "omega-3",
    "fish-oil": "omega-3",
    "fish-oil-omega3": "omega-3",
    "vitamin-d3": "vitamin-d",
    "vit-d": "vitamin-d",
    "vitd": "vitamin-d",
    "mag": "magnesium",
    "magnesium-glycinate": "magnesium",
    "probiotic": "probiotics",
    "lactobacillus": "probiotics",
    "bifidobacterium": "probiotics",
    "l-theanine": "l-theanine",
    "theanine": "l-theanine",
    "s-adenosyl-methionine": "same",
    "ademetionine": "same",
    "melatonin": "melatonin",
    "coffee": "caffeine",
    "espresso": "caffeine",
    "rhodiola-rosea": "rhodiola",
    "golden-root": "rhodiola",
    "withania-somnifera": "ashwagandha",
    "indian-ginseng": "ashwagandha",
    "ksm-66": "ashwagandha",
}

# Lower-is-better metrics (more negative = better outcome)
_LOWER_IS_BETTER = {"stress_index"}


def _resolve_canonical(name: str) -> Optional[str]:
    """Resolve a supplement name to its canonical DB key."""
    key = name.lower().strip().replace(" ", "-").replace("_", "-")
    if key in SUPPLEMENT_DB:
        return key
    mapped = _ALIASES.get(key)
    if mapped:
        return mapped
    for db_key in SUPPLEMENT_DB:
        if db_key.startswith(key) or key in db_key:
            return db_key
    return None


def lookup(name: str) -> Optional[dict]:
    """Return knowledge base entry for a supplement name.

    Does fuzzy matching on common aliases (e.g. 'omega3' → 'omega-3').
    """
    canonical = _resolve_canonical(name)
    if canonical:
        return SUPPLEMENT_DB[canonical]
    return None


def get_supplement_knowledge(name: str) -> Optional[dict]:
    """Return knowledge base entry with canonical_name included."""
    canonical = _resolve_canonical(name)
    if canonical is None:
        return None
    entry = dict(SUPPLEMENT_DB[canonical])
    entry["canonical_name"] = canonical
    return entry


def check_interactions(supplement_names: List[str]) -> dict:
    """Return interaction warnings/synergies for a set of supplements.

    Returns a dict with canonical_names, interaction_count, and interactions list.
    """
    canonical_names = []
    canonical_set: set = set()
    for n in supplement_names:
        resolved = _resolve_canonical(n)
        canon = resolved if resolved else n.lower().strip().replace(" ", "-").replace("_", "-")
        canonical_names.append(canon)
        canonical_set.add(canon)

    interactions = []
    for rule in INTERACTION_RULES:
        if all(s in canonical_set for s in rule["supplements"]):
            interactions.append(rule)

    return {
        "canonical_names": canonical_names,
        "interaction_count": len(interactions),
        "interactions": interactions,
        "has_cautions": any(i["type"] == "caution" for i in interactions),
        "has_synergies": any(i["type"] == "synergy" for i in interactions),
    }


def population_vs_personal(
    personal_effects: dict,
    entry: Optional[dict] = None,
    *,
    supplement_name: Optional[str] = None,
) -> dict:
    """Compare personal observed effects vs population-average expected effects.

    Args:
        personal_effects: dict with observed shifts (avg_valence_shift, avg_stress_shift,
            avg_focus_shift) or (valence, stress_index, focus_index).
        entry: optional knowledge-base entry dict (with expected_effects).
            If omitted, looked up via supplement_name.
        supplement_name: supplement identifier (used when entry is not given).

    Returns:
        dict with metrics list containing per-metric comparisons.
    """
    if entry is None:
        if supplement_name is None:
            return {"metrics": []}
        entry = lookup(supplement_name)
        if entry is None:
            return {"metrics": []}

    expected = entry.get("expected_effects", {})

    # Map personal effect keys to expected effect keys
    key_map = {
        "avg_valence_shift": "valence",
        "avg_stress_shift": "stress_index",
        "avg_focus_shift": "focus_index",
    }

    metrics = []
    for key, exp_val in expected.items():
        # Try direct key first, then mapped keys
        personal_val = personal_effects.get(key)
        if personal_val is None:
            for alt_key, mapped_key in key_map.items():
                if mapped_key == key:
                    personal_val = personal_effects.get(alt_key)
                    break
        if personal_val is None:
            continue

        diff = personal_val - exp_val

        if key in _LOWER_IS_BETTER:
            # For stress: more negative personal = better = above_average
            if abs(diff) < 0.02:
                direction = "average"
            elif diff < 0:
                direction = "above_average"
            else:
                direction = "below_average"
        else:
            if abs(diff) < 0.02:
                direction = "average"
            elif diff > 0:
                direction = "above_average"
            else:
                direction = "below_average"

        metrics.append({
            "metric": key,
            "personal": personal_val,
            "population_avg": exp_val,
            "diff": round(diff, 3),
            "comparison": direction,
        })

    return {"metrics": metrics}
