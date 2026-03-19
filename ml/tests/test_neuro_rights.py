"""Tests for neuro-rights governance framework.

Covers:
  - Consent ledger: grant, revoke, immutability hashing, filtering
  - Data inventory: registration, access flows, sovereignty dashboard
  - Rights audit: all five neuro-rights, violations, compliance
  - Deletion impact: GDPR Art 17 analysis
  - Explanation generation: plain-language ML prediction explanations
  - Data minimization: over-collection flagging
  - Governance report: comprehensive report assembly
  - API routes: POST /audit, POST /consent, GET /inventory, GET /status
"""
import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.neuro_rights import (
    CONSENT_ACTIONS,
    NEURAL_DATA_TYPES,
    NEURO_RIGHTS_DESCRIPTIONS,
    NeuroRight,
    _reset_stores,
    audit_rights_compliance,
    check_consent,
    check_data_minimization,
    compute_data_inventory,
    compute_deletion_impact,
    compute_governance_report,
    generate_explanation,
    get_consent_ledger,
    log_consent,
    register_data_flow,
    register_data_record,
    report_to_dict,
)


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_stores():
    """Reset all in-memory stores before and after every test."""
    _reset_stores()
    yield
    _reset_stores()


@pytest.fixture
def user_id():
    return "user-001"


@pytest.fixture
def app():
    """Create a FastAPI app with the neuro-rights router mounted."""
    from api.routes.neuro_rights import router
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app)


# -- NeuroRight enum ---------------------------------------------------------

class TestNeuroRightEnum:
    def test_all_five_rights_exist(self):
        rights = list(NeuroRight)
        assert len(rights) == 5

    def test_enum_values(self):
        assert NeuroRight.MENTAL_PRIVACY.value == "mental_privacy"
        assert NeuroRight.COGNITIVE_LIBERTY.value == "cognitive_liberty"
        assert NeuroRight.MENTAL_INTEGRITY.value == "mental_integrity"
        assert NeuroRight.PSYCHOLOGICAL_CONTINUITY.value == "psychological_continuity"
        assert NeuroRight.FAIR_ACCESS.value == "fair_access"

    def test_all_rights_have_descriptions(self):
        for right in NeuroRight:
            assert right in NEURO_RIGHTS_DESCRIPTIONS
            assert len(NEURO_RIGHTS_DESCRIPTIONS[right]) > 10


# -- Consent ledger ----------------------------------------------------------

class TestConsentLedger:
    def test_grant_consent(self, user_id):
        entry = log_consent(user_id, "eeg_raw", "grant")
        assert entry["user_id"] == user_id
        assert entry["data_type"] == "eeg_raw"
        assert entry["action"] == "grant"
        assert "hash" in entry
        assert "timestamp" in entry
        assert "id" in entry

    def test_revoke_consent(self, user_id):
        log_consent(user_id, "eeg_raw", "grant")
        entry = log_consent(user_id, "eeg_raw", "revoke")
        assert entry["action"] == "revoke"
        assert check_consent(user_id, "eeg_raw") is False

    def test_consent_state_tracks_latest(self, user_id):
        log_consent(user_id, "eeg_raw", "grant")
        assert check_consent(user_id, "eeg_raw") is True
        log_consent(user_id, "eeg_raw", "revoke")
        assert check_consent(user_id, "eeg_raw") is False
        log_consent(user_id, "eeg_raw", "grant")
        assert check_consent(user_id, "eeg_raw") is True

    def test_ledger_is_append_only(self, user_id):
        log_consent(user_id, "eeg_raw", "grant")
        log_consent(user_id, "eeg_raw", "revoke")
        log_consent(user_id, "eeg_raw", "grant")

        ledger = get_consent_ledger(user_id)
        assert len(ledger) == 3
        assert ledger[0]["action"] == "grant"
        assert ledger[1]["action"] == "revoke"
        assert ledger[2]["action"] == "grant"

    def test_hash_uniqueness(self, user_id):
        e1 = log_consent(user_id, "eeg_raw", "grant")
        e2 = log_consent(user_id, "eeg_features", "grant")
        assert e1["hash"] != e2["hash"]

    def test_invalid_action_raises(self, user_id):
        with pytest.raises(ValueError, match="Invalid action"):
            log_consent(user_id, "eeg_raw", "delete")

    def test_invalid_data_type_raises(self, user_id):
        with pytest.raises(ValueError, match="Unknown data type"):
            log_consent(user_id, "invalid_type", "grant")

    def test_filter_by_user(self):
        log_consent("user-a", "eeg_raw", "grant")
        log_consent("user-b", "eeg_raw", "grant")
        assert len(get_consent_ledger("user-a")) == 1
        assert len(get_consent_ledger("user-b")) == 1
        assert len(get_consent_ledger()) == 2


# -- Data inventory ----------------------------------------------------------

class TestDataInventory:
    def test_register_record(self, user_id):
        record = register_data_record(user_id, "eeg_raw", source="muse2")
        assert record["user_id"] == user_id
        assert record["data_type"] == "eeg_raw"
        assert record["source"] == "muse2"
        assert "id" in record
        assert "created_at" in record

    def test_register_invalid_type_raises(self, user_id):
        with pytest.raises(ValueError, match="Unknown data type"):
            register_data_record(user_id, "invalid_type")

    def test_compute_inventory(self, user_id):
        register_data_record(user_id, "eeg_raw")
        register_data_record(user_id, "emotion_predictions")
        register_data_flow(user_id, "eeg_raw", accessor="model_a")

        inventory = compute_data_inventory(user_id)
        assert inventory["user_id"] == user_id
        assert inventory["total_records"] == 2
        assert inventory["total_access_events"] == 1
        assert "data_types" in inventory
        assert len(inventory["data_types"]) == len(NEURAL_DATA_TYPES)

    def test_empty_inventory(self):
        inventory = compute_data_inventory("nonexistent-user")
        assert inventory["total_records"] == 0
        assert inventory["total_access_events"] == 0


# -- Rights audit ------------------------------------------------------------

class TestRightsAudit:
    def test_fully_compliant(self, user_id):
        """With no data flows and no inventory, everything is compliant."""
        result = audit_rights_compliance(user_id)
        assert result["overall_compliant"] is True
        assert result["total_violations"] == 0
        assert len(result["rights"]) == 5

    def test_mental_privacy_violation(self, user_id):
        """Accessing data without consent triggers a mental privacy violation."""
        register_data_flow(user_id, "eeg_raw", accessor="third_party")
        # No consent granted for eeg_raw

        result = audit_rights_compliance(user_id)
        privacy_status = result["rights"]["mental_privacy"]
        assert privacy_status["compliant"] is False
        assert len(privacy_status["issues"]) == 1
        assert privacy_status["issues"][0]["type"] == "unconsented_access"

    def test_mental_privacy_ok_with_consent(self, user_id):
        """Accessing data WITH consent should not trigger a violation."""
        log_consent(user_id, "eeg_raw", "grant")
        register_data_flow(user_id, "eeg_raw", accessor="model_a")

        result = audit_rights_compliance(user_id)
        privacy_status = result["rights"]["mental_privacy"]
        assert privacy_status["compliant"] is True

    def test_cognitive_liberty_violation(self, user_id):
        """External entity controlling neurofeedback triggers a violation."""
        register_data_flow(
            user_id, "neurofeedback_protocols",
            accessor="external_corp", purpose="analysis",
        )

        result = audit_rights_compliance(user_id)
        liberty_status = result["rights"]["cognitive_liberty"]
        assert liberty_status["compliant"] is False
        assert liberty_status["issues"][0]["type"] == "external_neurofeedback"

    def test_mental_integrity_violation(self, user_id):
        """Unauthorized modification of neural data triggers a violation."""
        register_data_flow(
            user_id, "eeg_raw",
            accessor="bad_actor", purpose="modification",
        )
        # No consent for eeg_raw

        result = audit_rights_compliance(user_id)
        integrity_status = result["rights"]["mental_integrity"]
        assert integrity_status["compliant"] is False

    def test_psychological_continuity_violation(self, user_id):
        """Unauthorized identity alteration triggers a violation."""
        register_data_flow(
            user_id, "emotion_predictions",
            accessor="system", purpose="alteration",
        )

        result = audit_rights_compliance(user_id)
        continuity_status = result["rights"]["psychological_continuity"]
        assert continuity_status["compliant"] is False

    def test_audit_with_explicit_data_flows(self, user_id):
        """Audit with externally provided data flows instead of stored ones."""
        explicit_flows = [
            {"data_type": "eeg_raw", "accessor": "model_x", "timestamp": "2026-01-01T00:00:00Z"},
        ]
        result = audit_rights_compliance(user_id, data_flows=explicit_flows)
        # eeg_raw has no consent -> violation
        assert result["overall_compliant"] is False

    def test_audit_counts_flows(self, user_id):
        register_data_flow(user_id, "eeg_raw", accessor="a")
        register_data_flow(user_id, "eeg_features", accessor="b")
        log_consent(user_id, "eeg_raw", "grant")
        log_consent(user_id, "eeg_features", "grant")

        result = audit_rights_compliance(user_id)
        assert result["data_flows_audited"] == 2


# -- Deletion impact ---------------------------------------------------------

class TestDeletionImpact:
    def test_deletion_impact_structure(self, user_id):
        register_data_record(user_id, "eeg_raw")
        register_data_record(user_id, "emotion_predictions")
        log_consent(user_id, "eeg_raw", "grant")

        result = compute_deletion_impact(user_id)
        assert result["user_id"] == user_id
        assert "deletion_summary" in result
        summary = result["deletion_summary"]
        assert summary["data_records"] == 2
        assert summary["consent_ledger_entries"] == 1
        assert "eeg_raw" in summary["affected_data_types"]
        assert len(summary["affected_ml_models"]) > 0
        assert "gdpr_article" in result

    def test_empty_deletion_impact(self):
        result = compute_deletion_impact("no-data-user")
        assert result["deletion_summary"]["data_records"] == 0
        assert result["estimated_records_total"] == 0


# -- Explanation generation --------------------------------------------------

class TestExplanation:
    def test_basic_explanation(self):
        prediction = {
            "emotion": "happy",
            "confidence": 0.85,
            "valence": 0.6,
            "arousal": 0.7,
        }
        result = generate_explanation(prediction, model_name="emotion_classifier")
        assert "emotion_classifier" in result["explanation"]
        assert "happy" in result["explanation"]
        assert "85.0%" in result["explanation"]
        assert result["model_name"] == "emotion_classifier"
        assert "data_rights_notice" in result

    def test_explanation_with_probabilities(self):
        prediction = {
            "emotion": "sad",
            "confidence": 0.6,
            "probabilities": {"sad": 0.6, "neutral": 0.2, "happy": 0.1, "angry": 0.1},
        }
        result = generate_explanation(prediction, model_name="test_model")
        assert "sad: 60.0%" in result["explanation"]

    def test_explanation_with_custom_input_summary(self):
        prediction = {"label": "N2"}
        result = generate_explanation(
            prediction,
            model_name="sleep_staging",
            input_summary="30-second EEG epoch from channel AF7",
        )
        assert "30-second EEG epoch" in result["explanation"]

    def test_explanation_with_faa(self):
        prediction = {
            "emotion": "happy",
            "confidence": 0.7,
            "frontal_asymmetry": 0.15,
            "stress_index": 0.3,
        }
        result = generate_explanation(prediction, model_name="emotion_classifier")
        assert len(result["key_factors"]) >= 2


# -- Data minimization -------------------------------------------------------

class TestDataMinimization:
    def test_no_flags_when_proportionate(self, user_id):
        register_data_record(user_id, "eeg_raw", purpose="emotion_analysis")
        register_data_record(user_id, "eeg_features", purpose="emotion_analysis")

        result = check_data_minimization(user_id)
        assert result["flagged_unnecessary"] == 0
        assert result["minimization_score"] == 1.0

    def test_flags_unnecessary_collection(self, user_id):
        register_data_record(user_id, "eeg_raw", purpose="emotion_analysis")
        register_data_record(user_id, "voice_biomarkers", purpose="emotion_analysis")

        result = check_data_minimization(user_id)
        assert result["flagged_unnecessary"] >= 1
        assert result["minimization_score"] < 1.0
        assert any(f["data_type"] == "voice_biomarkers" for f in result["flags"])

    def test_empty_minimization(self):
        result = check_data_minimization("empty-user")
        assert result["total_data_types_collected"] == 0
        assert result["flagged_unnecessary"] == 0


# -- Governance report -------------------------------------------------------

class TestGovernanceReport:
    def test_report_structure(self, user_id):
        register_data_record(user_id, "eeg_raw")
        log_consent(user_id, "eeg_raw", "grant")

        report = compute_governance_report(user_id)
        assert report["user_id"] == user_id
        assert "rights_audit" in report
        assert "data_inventory" in report
        assert "data_minimization" in report
        assert "deletion_impact" in report
        assert "consent_history" in report
        assert "neuro_rights_reference" in report

    def test_report_to_dict_serializes_enums(self, user_id):
        report = compute_governance_report(user_id)
        serialized = report_to_dict(report)
        # Should be JSON-safe: no Enum instances remain
        import json
        json_str = json.dumps(serialized)
        assert "mental_privacy" in json_str


# -- API routes --------------------------------------------------------------

class TestAPIRoutes:
    def test_status_endpoint(self, client):
        resp = client.get("/neuro-rights/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "available"
        assert len(data["rights"]) == 5
        assert "eeg_raw" in data["supported_data_types"]

    def test_consent_endpoint(self, client):
        resp = client.post("/neuro-rights/consent", json={
            "user_id": "test-user",
            "data_type": "eeg_raw",
            "action": "grant",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["consent_entry"]["action"] == "grant"

    def test_consent_invalid_action(self, client):
        resp = client.post("/neuro-rights/consent", json={
            "user_id": "test-user",
            "data_type": "eeg_raw",
            "action": "delete",
        })
        assert resp.status_code == 400

    def test_consent_invalid_data_type(self, client):
        resp = client.post("/neuro-rights/consent", json={
            "user_id": "test-user",
            "data_type": "invalid_type",
            "action": "grant",
        })
        assert resp.status_code == 400

    def test_audit_endpoint(self, client):
        resp = client.post("/neuro-rights/audit", json={
            "user_id": "test-user",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_compliant" in data
        assert "rights" in data

    def test_audit_with_explicit_flows(self, client):
        resp = client.post("/neuro-rights/audit", json={
            "user_id": "test-user",
            "data_flows": [
                {"data_type": "eeg_raw", "accessor": "third_party"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        # eeg_raw without consent -> not compliant
        assert data["overall_compliant"] is False

    def test_inventory_endpoint(self, client):
        resp = client.get("/neuro-rights/inventory/test-user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "test-user"
        assert "total_records" in data
        assert "data_types" in data
