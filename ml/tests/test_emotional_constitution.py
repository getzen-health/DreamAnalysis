"""Tests for emotional constitution -- user-authored sovereignty framework.

Covers:
  - Constitution creation: happy path, duplicate rejection
  - Articles: add, validate domain/effect/priority, conditions
  - Amendments: update fields, preserve history, version tracking
  - Rule engine: evaluate single action, batch compliance, no-constitution default
  - Conflict resolution: priority ordering, recency tiebreak, safety-first deny
  - Profile: compute profile, coverage tracking, serialization
  - API routes: all five endpoints including error paths
"""
import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.emotional_constitution import (
    DOMAIN_DESCRIPTIONS,
    ArticleDomain,
    ActionType,
    ComplianceVerdict,
    DEFAULT_PRIORITY,
    _reset_stores,
    add_article,
    amend_article,
    check_compliance,
    compute_constitution_profile,
    create_constitution,
    evaluate_action,
    get_constitution,
    get_constitution_history,
    profile_to_dict,
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
def constitution(user_id):
    """Create a constitution and return it."""
    return create_constitution(user_id)


@pytest.fixture
def app():
    """Create a FastAPI app with the emotional constitution router mounted."""
    from api.routes.emotional_constitution import router
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app)


# -- Constitution creation ---------------------------------------------------

class TestCreateConstitution:
    def test_create_returns_structure(self, user_id):
        c = create_constitution(user_id)
        assert c["user_id"] == user_id
        assert "id" in c
        assert c["version"] == 1
        assert c["articles"] == {}
        assert "created_at" in c
        assert "preamble" in c

    def test_create_custom_name_and_preamble(self, user_id):
        c = create_constitution(
            user_id,
            name="My Rules",
            preamble="I control my data.",
        )
        assert c["name"] == "My Rules"
        assert c["preamble"] == "I control my data."

    def test_duplicate_raises(self, user_id, constitution):
        with pytest.raises(ValueError, match="already has a constitution"):
            create_constitution(user_id)

    def test_get_constitution(self, user_id, constitution):
        result = get_constitution(user_id)
        assert result is not None
        assert result["user_id"] == user_id

    def test_get_nonexistent_returns_none(self):
        assert get_constitution("no-such-user") is None


# -- Articles ----------------------------------------------------------------

class TestArticles:
    def test_add_article(self, user_id, constitution):
        article = add_article(
            user_id,
            domain="data_sharing_rules",
            title="No third-party sharing",
            rule="Emotional data must never be shared with third parties.",
            effect="deny",
        )
        assert article["domain"] == "data_sharing_rules"
        assert article["title"] == "No third-party sharing"
        assert article["effect"] == "deny"
        assert article["priority"] == DEFAULT_PRIORITY
        assert article["version"] == 1
        assert "id" in article

    def test_add_article_increments_constitution_version(self, user_id, constitution):
        add_article(
            user_id,
            domain="privacy_red_lines",
            title="Test",
            rule="Test rule",
        )
        c = get_constitution(user_id)
        assert c["version"] == 2

    def test_invalid_domain_raises(self, user_id, constitution):
        with pytest.raises(ValueError, match="Invalid domain"):
            add_article(
                user_id,
                domain="bogus_domain",
                title="Test",
                rule="Test",
            )

    def test_invalid_effect_raises(self, user_id, constitution):
        with pytest.raises(ValueError, match="Invalid effect"):
            add_article(
                user_id,
                domain="data_sharing_rules",
                title="Test",
                rule="Test",
                effect="obliterate",
            )

    def test_invalid_priority_raises(self, user_id, constitution):
        with pytest.raises(ValueError, match="Priority must be"):
            add_article(
                user_id,
                domain="data_sharing_rules",
                title="Test",
                rule="Test",
                priority=200,
            )

    def test_no_constitution_raises(self):
        with pytest.raises(ValueError, match="No constitution found"):
            add_article(
                "ghost-user",
                domain="data_sharing_rules",
                title="Test",
                rule="Test",
            )

    def test_add_article_with_conditions(self, user_id, constitution):
        article = add_article(
            user_id,
            domain="data_sharing_rules",
            title="Limit emotion sharing",
            rule="Only share when recipient is therapist",
            action_types=["share_data"],
            conditions={"data_type": "emotion_predictions"},
            effect="deny",
        )
        assert article["action_types"] == ["share_data"]
        assert article["conditions"] == {"data_type": "emotion_predictions"}


# -- Amendments --------------------------------------------------------------

class TestAmendments:
    def test_amend_updates_fields(self, user_id, constitution):
        article = add_article(
            user_id,
            domain="ai_behavior_boundaries",
            title="Original title",
            rule="Original rule",
            priority=30,
        )
        amended = amend_article(
            user_id,
            article["id"],
            title="Updated title",
            rule="Updated rule",
            priority=80,
            reason="Strengthened the boundary",
        )
        assert amended["title"] == "Updated title"
        assert amended["rule"] == "Updated rule"
        assert amended["priority"] == 80
        assert amended["version"] == 2

    def test_amend_preserves_history(self, user_id, constitution):
        article = add_article(
            user_id,
            domain="crisis_protocols",
            title="Crisis rule",
            rule="V1",
        )
        amend_article(user_id, article["id"], rule="V2", reason="Revision 1")
        amend_article(user_id, article["id"], rule="V3", reason="Revision 2")

        history = get_constitution_history(user_id)
        assert len(history) == 2
        assert history[0]["old_state"]["rule"] == "V1"
        assert history[0]["new_state"]["rule"] == "V2"
        assert history[0]["reason"] == "Revision 1"
        assert history[1]["old_state"]["rule"] == "V2"
        assert history[1]["new_state"]["rule"] == "V3"

    def test_amend_nonexistent_article_raises(self, user_id, constitution):
        with pytest.raises(ValueError, match="not found"):
            amend_article(user_id, "fake-id", title="X")

    def test_amend_no_constitution_raises(self):
        with pytest.raises(ValueError, match="No constitution found"):
            amend_article("ghost-user", "fake-id", title="X")


# -- Rule engine: evaluate_action --------------------------------------------

class TestEvaluateAction:
    def test_no_constitution_allows_by_default(self):
        result = evaluate_action("no-user", "share_data")
        assert result["verdict"] == "allowed"
        assert "No constitution exists" in result["reason"]

    def test_no_matching_articles_allows(self, user_id, constitution):
        result = evaluate_action(user_id, "share_data")
        assert result["verdict"] == "allowed"

    def test_deny_article_blocks_action(self, user_id, constitution):
        add_article(
            user_id,
            domain="data_sharing_rules",
            title="Block all sharing",
            rule="Never share",
            action_types=["share_data"],
            effect="deny",
        )
        result = evaluate_action(user_id, "share_data")
        assert result["verdict"] == "denied"
        assert len(result["matched_articles"]) == 1

    def test_allow_article_permits_action(self, user_id, constitution):
        add_article(
            user_id,
            domain="intervention_preferences",
            title="Allow interventions",
            rule="System may intervene",
            action_types=["trigger_intervention"],
            effect="allow",
        )
        result = evaluate_action(user_id, "trigger_intervention")
        assert result["verdict"] == "allowed"

    def test_conditional_article(self, user_id, constitution):
        add_article(
            user_id,
            domain="data_sharing_rules",
            title="Conditional sharing",
            rule="Needs review",
            action_types=["share_data"],
            effect="conditional",
        )
        result = evaluate_action(user_id, "share_data")
        assert result["verdict"] == "conditional"

    def test_condition_matching(self, user_id, constitution):
        add_article(
            user_id,
            domain="privacy_red_lines",
            title="Block emotion export",
            rule="Never export emotion data",
            action_types=["export_data"],
            conditions={"data_type": "emotion_predictions"},
            effect="deny",
        )
        # Matching context -> denied
        result = evaluate_action(
            user_id, "export_data",
            context={"data_type": "emotion_predictions"},
        )
        assert result["verdict"] == "denied"

        # Non-matching context -> allowed (article doesn't match)
        result = evaluate_action(
            user_id, "export_data",
            context={"data_type": "sleep_staging"},
        )
        assert result["verdict"] == "allowed"

    def test_list_condition_matching(self, user_id, constitution):
        add_article(
            user_id,
            domain="data_sharing_rules",
            title="Allow therapist only",
            rule="Only therapist and doctor",
            action_types=["share_data"],
            conditions={"recipient": ["therapist", "doctor"]},
            effect="allow",
        )
        # Matching value in list -> matches
        result = evaluate_action(
            user_id, "share_data",
            context={"recipient": "therapist"},
        )
        assert result["verdict"] == "allowed"

        # Non-matching value -> article doesn't match -> default allow
        result = evaluate_action(
            user_id, "share_data",
            context={"recipient": "advertiser"},
        )
        assert result["verdict"] == "allowed"


# -- Conflict resolution -----------------------------------------------------

class TestConflictResolution:
    def test_higher_priority_wins(self, user_id, constitution):
        add_article(
            user_id,
            domain="data_sharing_rules",
            title="Low priority allow",
            rule="Allow sharing",
            action_types=["share_data"],
            effect="allow",
            priority=30,
        )
        add_article(
            user_id,
            domain="privacy_red_lines",
            title="High priority deny",
            rule="Never share",
            action_types=["share_data"],
            effect="deny",
            priority=90,
        )
        result = evaluate_action(user_id, "share_data")
        assert result["verdict"] == "denied"

    def test_same_priority_latest_wins(self, user_id, constitution):
        a1 = add_article(
            user_id,
            domain="data_sharing_rules",
            title="Earlier deny",
            rule="Deny",
            action_types=["share_data"],
            effect="deny",
            priority=50,
        )
        # Amend a1 to set it to allow at same priority, making it newer
        amend_article(user_id, a1["id"], effect="allow", reason="Changed mind")
        result = evaluate_action(user_id, "share_data")
        assert result["verdict"] == "allowed"


# -- Batch compliance --------------------------------------------------------

class TestBatchCompliance:
    def test_batch_all_allowed(self, user_id, constitution):
        result = check_compliance(user_id, [
            {"action_type": "share_data"},
            {"action_type": "store_reading"},
        ])
        assert result["all_compliant"] is True
        assert result["denied_count"] == 0
        assert result["total_actions"] == 2

    def test_batch_with_denial(self, user_id, constitution):
        add_article(
            user_id,
            domain="privacy_red_lines",
            title="Block notify",
            rule="No third party notifications",
            action_types=["notify_third_party"],
            effect="deny",
        )
        result = check_compliance(user_id, [
            {"action_type": "store_reading"},
            {"action_type": "notify_third_party"},
        ])
        assert result["all_compliant"] is False
        assert result["denied_count"] == 1
        assert len(result["results"]) == 2


# -- Profile and serialization -----------------------------------------------

class TestProfile:
    def test_no_constitution_profile(self):
        profile = compute_constitution_profile("ghost-user")
        assert profile["exists"] is False

    def test_profile_with_articles(self, user_id, constitution):
        add_article(
            user_id,
            domain="data_sharing_rules",
            title="Rule A",
            rule="A",
            action_types=["share_data"],
            priority=70,
        )
        add_article(
            user_id,
            domain="data_sharing_rules",
            title="Rule B",
            rule="B",
            action_types=["export_data"],
            priority=30,
        )
        add_article(
            user_id,
            domain="crisis_protocols",
            title="Crisis",
            rule="C",
            action_types=["notify_third_party"],
        )

        profile = compute_constitution_profile(user_id)
        assert profile["exists"] is True
        assert profile["total_articles"] == 3
        assert profile["articles_per_domain"]["data_sharing_rules"] == 2
        assert profile["articles_per_domain"]["crisis_protocols"] == 1
        assert "share_data" in profile["covered_action_types"]
        assert profile["coverage_ratio"] > 0

    def test_profile_to_dict_serializable(self, user_id, constitution):
        add_article(
            user_id,
            domain="ai_behavior_boundaries",
            title="Limit",
            rule="Test",
        )
        profile = compute_constitution_profile(user_id)
        serialized = profile_to_dict(profile)
        import json
        json_str = json.dumps(serialized)
        assert "ai_behavior_boundaries" in json_str


# -- API routes --------------------------------------------------------------

class TestAPIRoutes:
    def test_status_endpoint(self, client):
        resp = client.get("/constitution/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "available"
        assert len(data["domains"]) == 5
        assert "share_data" in data["action_types"]

    def test_create_endpoint(self, client):
        resp = client.post("/constitution/create", json={
            "user_id": "test-user",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert data["constitution"]["user_id"] == "test-user"

    def test_create_duplicate_returns_400(self, client):
        client.post("/constitution/create", json={"user_id": "test-user"})
        resp = client.post("/constitution/create", json={"user_id": "test-user"})
        assert resp.status_code == 400

    def test_add_article_endpoint(self, client):
        client.post("/constitution/create", json={"user_id": "test-user"})
        resp = client.post("/constitution/article", json={
            "user_id": "test-user",
            "domain": "privacy_red_lines",
            "title": "No selling data",
            "rule": "My emotional data must never be sold.",
            "effect": "deny",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "added"
        assert data["article"]["title"] == "No selling data"

    def test_amend_article_endpoint(self, client):
        client.post("/constitution/create", json={"user_id": "test-user"})
        add_resp = client.post("/constitution/article", json={
            "user_id": "test-user",
            "domain": "data_sharing_rules",
            "title": "Original",
            "rule": "Original rule",
        })
        article_id = add_resp.json()["article"]["id"]

        resp = client.post("/constitution/article", json={
            "user_id": "test-user",
            "article_id": article_id,
            "domain": "data_sharing_rules",
            "title": "Amended",
            "rule": "Amended rule",
            "reason": "Policy update",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "amended"
        assert data["article"]["title"] == "Amended"

    def test_evaluate_endpoint(self, client):
        client.post("/constitution/create", json={"user_id": "test-user"})
        client.post("/constitution/article", json={
            "user_id": "test-user",
            "domain": "privacy_red_lines",
            "title": "Block export",
            "rule": "No exporting",
            "action_types": ["export_data"],
            "effect": "deny",
        })
        resp = client.post("/constitution/evaluate", json={
            "user_id": "test-user",
            "action_type": "export_data",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "denied"

    def test_get_constitution_endpoint(self, client):
        client.post("/constitution/create", json={"user_id": "test-user"})
        resp = client.get("/constitution/test-user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["constitution"]["user_id"] == "test-user"
        assert "profile" in data
        assert "amendment_history" in data

    def test_get_nonexistent_constitution_returns_404(self, client):
        resp = client.get("/constitution/no-such-user")
        assert resp.status_code == 404

    def test_article_no_constitution_returns_400(self, client):
        resp = client.post("/constitution/article", json={
            "user_id": "no-such-user",
            "domain": "data_sharing_rules",
            "title": "Test",
            "rule": "Test",
        })
        assert resp.status_code == 400
