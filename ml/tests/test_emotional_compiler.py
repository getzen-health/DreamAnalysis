"""Tests for the Emotional Compiler (issue #454).

Covers: translate_emotion, compile_across_frameworks, get_framework_vocabulary,
suggest_intervention_per_framework, compute_framework_profile, profile_to_dict,
framework data integrity, bidirectional mapping consistency, and API route
integration.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.emotional_compiler import (
    FRAMEWORKS,
    FRAMEWORK_DESCRIPTIONS,
    FRAMEWORK_VOCABULARY,
    INTERVENTIONS,
    _MAPPING_TABLE,
    compile_across_frameworks,
    compute_framework_profile,
    get_framework_vocabulary,
    profile_to_dict,
    suggest_intervention_per_framework,
    translate_emotion,
)


# ---------------------------------------------------------------------------
# 1. translate_emotion — core translation
# ---------------------------------------------------------------------------


def test_translate_cbt_to_act():
    """CBT catastrophizing should translate to ACT cognitive_fusion."""
    result = translate_emotion("CBT", "catastrophizing", "ACT")
    assert "error" not in result
    assert result["source_framework"] == "CBT"
    assert result["source_concept"] == "catastrophizing"
    assert result["target_framework"] == "ACT"
    assert result["target_concept"] == "cognitive_fusion"
    assert len(result["target_description"]) > 0


def test_translate_dbt_to_ifs():
    """DBT distress_intolerance should translate to IFS firefighter_response."""
    result = translate_emotion("DBT", "distress_intolerance", "IFS")
    assert "error" not in result
    assert result["target_concept"] == "firefighter_response"


def test_translate_somatic_to_cbt():
    """Somatic chest_tightness should translate to CBT catastrophizing."""
    result = translate_emotion("Somatic", "chest_tightness", "CBT")
    assert "error" not in result
    assert result["target_concept"] == "catastrophizing"


def test_translate_ifs_to_dbt():
    """IFS blended_state should translate to DBT emotion_mind."""
    result = translate_emotion("IFS", "blended_state", "DBT")
    assert "error" not in result
    assert result["target_concept"] == "emotion_mind"


def test_translate_includes_intervention():
    """Translation result must include a non-empty intervention for the target."""
    result = translate_emotion("CBT", "catastrophizing", "DBT")
    assert "error" not in result
    assert len(result["target_intervention"]) > 0


def test_translate_same_framework():
    """Translating within the same framework returns the same concept."""
    result = translate_emotion("CBT", "catastrophizing", "CBT")
    assert "error" not in result
    assert result["target_concept"] == "catastrophizing"


def test_translate_unknown_source_framework():
    """Unknown source framework should return error."""
    result = translate_emotion("Freudian", "catastrophizing", "CBT")
    assert "error" in result
    assert "Unknown source framework" in result["error"]


def test_translate_unknown_target_framework():
    """Unknown target framework should return error."""
    result = translate_emotion("CBT", "catastrophizing", "Jungian")
    assert "error" in result
    assert "Unknown target framework" in result["error"]


def test_translate_unknown_concept():
    """Unknown concept in valid framework should return error."""
    result = translate_emotion("CBT", "nonexistent_concept", "ACT")
    assert "error" in result
    assert "Unknown concept" in result["error"]


def test_translate_case_insensitive_framework():
    """Framework names should be case-insensitive."""
    result = translate_emotion("cbt", "catastrophizing", "act")
    assert "error" not in result
    assert result["source_framework"] == "CBT"
    assert result["target_framework"] == "ACT"


# ---------------------------------------------------------------------------
# 2. compile_across_frameworks
# ---------------------------------------------------------------------------


def test_compile_returns_all_frameworks():
    """Compilation must include translations for all 5 frameworks."""
    result = compile_across_frameworks("CBT", "catastrophizing")
    assert "error" not in result
    assert set(result["translations"].keys()) == set(FRAMEWORKS)
    assert set(result["descriptions"].keys()) == set(FRAMEWORKS)
    assert set(result["interventions"].keys()) == set(FRAMEWORKS)


def test_compile_source_is_identity():
    """Source framework translation should map to the same concept."""
    result = compile_across_frameworks("ACT", "cognitive_fusion")
    assert "error" not in result
    assert result["translations"]["ACT"] == "cognitive_fusion"


def test_compile_unknown_framework():
    """Compilation with unknown framework should return error."""
    result = compile_across_frameworks("NLP", "rapport")
    assert "error" in result


def test_compile_unknown_concept():
    """Compilation with unknown concept should return error."""
    result = compile_across_frameworks("DBT", "nonexistent")
    assert "error" in result


# ---------------------------------------------------------------------------
# 3. get_framework_vocabulary
# ---------------------------------------------------------------------------


def test_vocabulary_returns_all_concepts():
    """Each framework vocabulary should have 10 concepts."""
    for fw in FRAMEWORKS:
        result = get_framework_vocabulary(fw)
        assert "error" not in result
        assert result["concept_count"] == 10
        assert len(result["vocabulary"]) == 10


def test_vocabulary_unknown_framework():
    """Unknown framework should return error."""
    result = get_framework_vocabulary("Gestalt")
    assert "error" in result


def test_vocabulary_includes_description():
    """Vocabulary result should include framework description."""
    result = get_framework_vocabulary("CBT")
    assert "error" not in result
    assert len(result["description"]) > 0
    assert "Cognitive Behavioral" in result["description"]


# ---------------------------------------------------------------------------
# 4. suggest_intervention_per_framework
# ---------------------------------------------------------------------------


def test_intervention_per_framework():
    """All 5 frameworks should suggest an intervention."""
    result = suggest_intervention_per_framework("CBT", "catastrophizing")
    assert "error" not in result
    assert set(result["interventions"].keys()) == set(FRAMEWORKS)
    for fw in FRAMEWORKS:
        assert len(result["interventions"][fw]) > 0


def test_intervention_error_propagation():
    """Invalid input should propagate error from compile."""
    result = suggest_intervention_per_framework("CBT", "nonexistent")
    assert "error" in result


# ---------------------------------------------------------------------------
# 5. compute_framework_profile + profile_to_dict
# ---------------------------------------------------------------------------


def test_compute_framework_profile_valid():
    """Valid input should return a FrameworkProfile."""
    profile = compute_framework_profile("DBT", "emotion_mind")
    assert profile is not None
    assert profile.source_framework == "DBT"
    assert profile.source_concept == "emotion_mind"
    assert len(profile.translations) == 5


def test_compute_framework_profile_invalid():
    """Invalid input should return None."""
    profile = compute_framework_profile("CBT", "nonexistent")
    assert profile is None


def test_profile_to_dict_valid():
    """profile_to_dict should serialize to a plain dict."""
    profile = compute_framework_profile("IFS", "protector_activated")
    d = profile_to_dict(profile)
    assert d is not None
    assert isinstance(d, dict)
    assert d["source_framework"] == "IFS"
    assert "translations" in d
    assert "descriptions" in d
    assert "interventions" in d


def test_profile_to_dict_none():
    """profile_to_dict(None) should return None."""
    assert profile_to_dict(None) is None


# ---------------------------------------------------------------------------
# 6. Data integrity
# ---------------------------------------------------------------------------


def test_all_frameworks_have_descriptions():
    """Every framework must have a description."""
    for fw in FRAMEWORKS:
        assert fw in FRAMEWORK_DESCRIPTIONS
        assert len(FRAMEWORK_DESCRIPTIONS[fw]) > 0


def test_all_frameworks_have_vocabulary():
    """Every framework must have a vocabulary dict."""
    for fw in FRAMEWORKS:
        assert fw in FRAMEWORK_VOCABULARY
        assert len(FRAMEWORK_VOCABULARY[fw]) > 0


def test_all_frameworks_have_interventions():
    """Every framework must have interventions for all its concepts."""
    for fw in FRAMEWORKS:
        vocab = FRAMEWORK_VOCABULARY[fw]
        interventions = INTERVENTIONS.get(fw, {})
        for concept in vocab:
            assert concept in interventions, (
                f"Missing intervention for {fw}:{concept}"
            )
            assert len(interventions[concept]) > 0


def test_mapping_table_keys_valid():
    """Every key in the mapping table should reference a valid framework:concept."""
    for key in _MAPPING_TABLE:
        fw, concept = key.split(":", 1)
        assert fw in FRAMEWORKS, f"Invalid framework in mapping key: {key}"
        assert concept in FRAMEWORK_VOCABULARY[fw], (
            f"Invalid concept in mapping key: {key}"
        )


def test_mapping_table_values_valid():
    """Every value in the mapping table should reference a valid concept in
    the target framework."""
    for key, mapping in _MAPPING_TABLE.items():
        for fw, concept in mapping.items():
            assert fw in FRAMEWORKS, (
                f"Invalid target framework in mapping {key}: {fw}"
            )
            assert concept in FRAMEWORK_VOCABULARY[fw], (
                f"Invalid target concept in mapping {key} -> {fw}:{concept}"
            )


def test_every_concept_has_mapping():
    """Every concept in every framework should appear in the mapping table."""
    for fw in FRAMEWORKS:
        for concept in FRAMEWORK_VOCABULARY[fw]:
            key = f"{fw}:{concept}"
            assert key in _MAPPING_TABLE, f"Missing mapping for {key}"


# ---------------------------------------------------------------------------
# 7. Bidirectional consistency
# ---------------------------------------------------------------------------


def test_self_mapping_is_identity():
    """Every mapping entry should map to itself within its own framework."""
    for key, mapping in _MAPPING_TABLE.items():
        fw, concept = key.split(":", 1)
        assert mapping[fw] == concept, (
            f"Self-mapping broken for {key}: expected {concept}, got {mapping[fw]}"
        )


# ---------------------------------------------------------------------------
# 8. API route integration
# ---------------------------------------------------------------------------


def test_api_status_endpoint():
    """GET /compiler/status should return ok."""
    from fastapi.testclient import TestClient
    from api.routes.emotional_compiler import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.get("/compiler/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["frameworks_loaded"] == 5
    assert data["total_concepts"] == 50  # 5 frameworks x 10 concepts


def test_api_frameworks_endpoint():
    """GET /compiler/frameworks should list all 5 frameworks."""
    from fastapi.testclient import TestClient
    from api.routes.emotional_compiler import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.get("/compiler/frameworks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_frameworks"] == 5
    names = [f["name"] for f in data["frameworks"]]
    assert set(names) == set(FRAMEWORKS)


def test_api_translate_endpoint():
    """POST /compiler/translate should translate between frameworks."""
    from fastapi.testclient import TestClient
    from api.routes.emotional_compiler import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.post("/compiler/translate", json={
        "source_framework": "CBT",
        "concept": "catastrophizing",
        "target_framework": "IFS",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_concept"] == "protector_activated"


def test_api_compile_endpoint():
    """POST /compiler/compile should return all framework translations."""
    from fastapi.testclient import TestClient
    from api.routes.emotional_compiler import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.post("/compiler/compile", json={
        "source_framework": "Somatic",
        "concept": "shallow_breathing",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "translations" in data
    assert "intervention_suggestions" in data
    assert len(data["translations"]) == 5


def test_api_translate_error():
    """POST /compiler/translate with invalid data should return error in body."""
    from fastapi.testclient import TestClient
    from api.routes.emotional_compiler import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.post("/compiler/translate", json={
        "source_framework": "CBT",
        "concept": "nonexistent",
        "target_framework": "ACT",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
