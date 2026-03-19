"""Emotional constitution -- user-authored sovereignty framework.

Allows users to define a personal constitution governing how AI should handle
their emotional data, interventions, and behavior.  Each constitution contains
articles organized into five domains:

  1. data_sharing_rules        -- who may access emotional data and when
  2. intervention_preferences  -- how/when the system may intervene
  3. ai_behavior_boundaries    -- what the AI must never do
  4. crisis_protocols          -- escalation rules during emotional crises
  5. privacy_red_lines         -- absolute limits that cannot be overridden

A lightweight rule engine evaluates proposed actions (share data, trigger
intervention, store a reading) against the user's constitution and returns
a compliance verdict with any violated articles cited.

Constitutional amendments preserve full version history so users can track
how their sovereignty preferences evolve over time.

Conflict resolution: when two articles in the same domain produce
contradictory verdicts, the article with the higher priority wins.
Within the same priority, the most recently amended article wins.

GitHub issue: #458
"""
from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class ArticleDomain(str, Enum):
    """The five domains a constitutional article may belong to."""
    DATA_SHARING_RULES = "data_sharing_rules"
    INTERVENTION_PREFERENCES = "intervention_preferences"
    AI_BEHAVIOR_BOUNDARIES = "ai_behavior_boundaries"
    CRISIS_PROTOCOLS = "crisis_protocols"
    PRIVACY_RED_LINES = "privacy_red_lines"


DOMAIN_DESCRIPTIONS: Dict[str, str] = {
    ArticleDomain.DATA_SHARING_RULES: (
        "Rules governing who may access emotional data and under what conditions."
    ),
    ArticleDomain.INTERVENTION_PREFERENCES: (
        "Preferences for when and how the system may intervene in the user's "
        "emotional state."
    ),
    ArticleDomain.AI_BEHAVIOR_BOUNDARIES: (
        "Hard limits on what the AI is permitted to do with emotional data "
        "and analysis."
    ),
    ArticleDomain.CRISIS_PROTOCOLS: (
        "Escalation and safety rules that activate during detected emotional crises."
    ),
    ArticleDomain.PRIVACY_RED_LINES: (
        "Absolute privacy constraints that cannot be overridden by any other rule."
    ),
}


class ActionType(str, Enum):
    """Categories of actions the system may propose."""
    SHARE_DATA = "share_data"
    STORE_READING = "store_reading"
    TRIGGER_INTERVENTION = "trigger_intervention"
    MODIFY_MODEL = "modify_model"
    NOTIFY_THIRD_PARTY = "notify_third_party"
    EXPORT_DATA = "export_data"


class ComplianceVerdict(str, Enum):
    """Outcome of evaluating an action against the constitution."""
    ALLOWED = "allowed"
    DENIED = "denied"
    CONDITIONAL = "conditional"


# Default priority for articles when none is specified
DEFAULT_PRIORITY = 50

# Valid priority range (higher = takes precedence in conflicts)
MIN_PRIORITY = 0
MAX_PRIORITY = 100


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

# user_id -> constitution dict
_constitutions: Dict[str, Dict[str, Any]] = {}

# user_id -> list of amendment history entries
_amendment_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)


def _reset_stores() -> None:
    """Reset all in-memory stores.  For testing only."""
    _constitutions.clear()
    _amendment_history.clear()


# ---------------------------------------------------------------------------
# Constitution CRUD
# ---------------------------------------------------------------------------

def create_constitution(
    user_id: str,
    *,
    name: Optional[str] = None,
    preamble: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new empty constitution for a user.

    Raises ValueError if the user already has a constitution.
    """
    if user_id in _constitutions:
        raise ValueError(f"User {user_id} already has a constitution")

    constitution: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "name": name or f"{user_id}'s Emotional Constitution",
        "preamble": preamble or (
            "This constitution defines the sovereign rules governing "
            "how AI systems may interact with my emotional data."
        ),
        "version": 1,
        "articles": {},  # article_id -> article dict
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    _constitutions[user_id] = constitution
    log.info("Constitution created for user %s", user_id)
    return deepcopy(constitution)


def get_constitution(user_id: str) -> Optional[Dict[str, Any]]:
    """Return the current constitution for a user, or None."""
    c = _constitutions.get(user_id)
    return deepcopy(c) if c is not None else None


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------

def add_article(
    user_id: str,
    *,
    domain: str,
    title: str,
    rule: str,
    action_types: Optional[List[str]] = None,
    conditions: Optional[Dict[str, Any]] = None,
    effect: str = "deny",
    priority: int = DEFAULT_PRIORITY,
) -> Dict[str, Any]:
    """Add an article to the user's constitution.

    Args:
        user_id: Owner of the constitution.
        domain: One of the five ArticleDomain values.
        title: Short human-readable title.
        rule: Plain-language description of the rule.
        action_types: Which ActionType values this article governs.
            If None/empty, the article applies to all action types.
        conditions: Optional key-value conditions that must be met for the
            rule to trigger (e.g. {"data_type": "emotion_predictions"}).
        effect: "deny" to block matching actions, "allow" to permit them,
            or "conditional" for actions that need further review.
        priority: 0-100.  Higher priority wins in conflicts.

    Raises:
        ValueError if the user has no constitution, domain is invalid, or
        priority is out of range.
    """
    constitution = _constitutions.get(user_id)
    if constitution is None:
        raise ValueError(f"No constitution found for user {user_id}")

    # Validate domain
    valid_domains = {d.value for d in ArticleDomain}
    if domain not in valid_domains:
        raise ValueError(
            f"Invalid domain: {domain}. Must be one of {sorted(valid_domains)}"
        )

    # Validate effect
    valid_effects = {"deny", "allow", "conditional"}
    if effect not in valid_effects:
        raise ValueError(
            f"Invalid effect: {effect}. Must be one of {sorted(valid_effects)}"
        )

    # Validate priority
    if not (MIN_PRIORITY <= priority <= MAX_PRIORITY):
        raise ValueError(
            f"Priority must be between {MIN_PRIORITY} and {MAX_PRIORITY}, "
            f"got {priority}"
        )

    article_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    article: Dict[str, Any] = {
        "id": article_id,
        "domain": domain,
        "title": title,
        "rule": rule,
        "action_types": action_types or [],
        "conditions": conditions or {},
        "effect": effect,
        "priority": priority,
        "version": 1,
        "created_at": now,
        "updated_at": now,
    }

    constitution["articles"][article_id] = article
    constitution["updated_at"] = now
    constitution["version"] += 1

    log.info(
        "Article '%s' added to constitution for user %s (domain=%s)",
        title, user_id, domain,
    )
    return deepcopy(article)


def amend_article(
    user_id: str,
    article_id: str,
    *,
    title: Optional[str] = None,
    rule: Optional[str] = None,
    action_types: Optional[List[str]] = None,
    conditions: Optional[Dict[str, Any]] = None,
    effect: Optional[str] = None,
    priority: Optional[int] = None,
    reason: str = "",
) -> Dict[str, Any]:
    """Amend an existing article, preserving history.

    Only the fields that are explicitly passed (not None) are updated.

    Args:
        user_id: Owner of the constitution.
        article_id: ID of the article to amend.
        reason: Human-readable reason for the amendment.

    Raises:
        ValueError if the constitution or article does not exist.
    """
    constitution = _constitutions.get(user_id)
    if constitution is None:
        raise ValueError(f"No constitution found for user {user_id}")

    article = constitution["articles"].get(article_id)
    if article is None:
        raise ValueError(
            f"Article {article_id} not found in constitution for user {user_id}"
        )

    # Snapshot the old state for history
    old_snapshot = deepcopy(article)

    now = datetime.now(timezone.utc).isoformat()

    if title is not None:
        article["title"] = title
    if rule is not None:
        article["rule"] = rule
    if action_types is not None:
        article["action_types"] = action_types
    if conditions is not None:
        article["conditions"] = conditions
    if effect is not None:
        valid_effects = {"deny", "allow", "conditional"}
        if effect not in valid_effects:
            raise ValueError(
                f"Invalid effect: {effect}. Must be one of {sorted(valid_effects)}"
            )
        article["effect"] = effect
    if priority is not None:
        if not (MIN_PRIORITY <= priority <= MAX_PRIORITY):
            raise ValueError(
                f"Priority must be between {MIN_PRIORITY} and {MAX_PRIORITY}, "
                f"got {priority}"
            )
        article["priority"] = priority

    article["version"] += 1
    article["updated_at"] = now
    constitution["updated_at"] = now
    constitution["version"] += 1

    # Record amendment in history
    amendment_record = {
        "id": str(uuid.uuid4()),
        "article_id": article_id,
        "old_state": old_snapshot,
        "new_state": deepcopy(article),
        "reason": reason,
        "timestamp": now,
    }
    _amendment_history[user_id].append(amendment_record)

    log.info(
        "Article %s amended (v%d) for user %s: %s",
        article_id, article["version"], user_id, reason or "(no reason given)",
    )
    return deepcopy(article)


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

def _article_matches_action(
    article: Dict[str, Any],
    action_type: str,
    context: Dict[str, Any],
) -> bool:
    """Check whether an article's scope matches the proposed action."""
    # Action type filter -- empty list means "applies to all"
    if article["action_types"] and action_type not in article["action_types"]:
        return False

    # Condition matching -- every condition key must match the context value
    for key, expected in article["conditions"].items():
        actual = context.get(key)
        if actual is None:
            return False
        # Support list-of-allowed-values
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False

    return True


def _resolve_conflicts(
    matching_articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """When multiple articles match, resolve conflicts.

    Strategy:
      1. Highest priority wins.
      2. Same priority -> most recently updated wins.
      3. "deny" takes precedence over "allow" at the same priority+time
         (safety-first default).
    """
    if not matching_articles:
        # No articles match -> default allow
        return {
            "verdict": ComplianceVerdict.ALLOWED.value,
            "deciding_article": None,
            "reason": "No constitutional articles govern this action.",
        }

    # Sort: primary by priority descending, secondary by updated_at descending,
    # tertiary: deny before allow (safety-first)
    effect_order = {"deny": 0, "conditional": 1, "allow": 2}

    def sort_key(a: Dict[str, Any]):
        return (
            -a["priority"],
            a["updated_at"],  # reversed below (we want latest first)
            effect_order.get(a["effect"], 1),
        )

    # Custom sort: we want highest priority, latest updated, deny first
    sorted_articles = sorted(
        matching_articles,
        key=lambda a: (
            -a["priority"],
            a["updated_at"],
        ),
        reverse=False,
    )

    # Among articles with the top priority, pick the latest updated
    top_priority = sorted_articles[0]["priority"]
    top_tier = [a for a in sorted_articles if a["priority"] == top_priority]

    # Sort top tier by updated_at descending
    top_tier.sort(key=lambda a: a["updated_at"], reverse=True)

    # Among same-timestamp articles, deny wins (safety-first)
    if len(top_tier) > 1:
        top_time = top_tier[0]["updated_at"]
        same_time = [a for a in top_tier if a["updated_at"] == top_time]
        if len(same_time) > 1:
            for a in same_time:
                if a["effect"] == "deny":
                    return {
                        "verdict": ComplianceVerdict.DENIED.value,
                        "deciding_article": a["id"],
                        "reason": f"Article '{a['title']}' denies this action (safety-first tiebreak).",
                    }

    winner = top_tier[0]
    verdict_map = {
        "deny": ComplianceVerdict.DENIED,
        "allow": ComplianceVerdict.ALLOWED,
        "conditional": ComplianceVerdict.CONDITIONAL,
    }

    return {
        "verdict": verdict_map[winner["effect"]].value,
        "deciding_article": winner["id"],
        "reason": f"Article '{winner['title']}' (priority {winner['priority']}) governs this action.",
    }


def evaluate_action(
    user_id: str,
    action_type: str,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate whether a proposed action complies with the user's constitution.

    Args:
        user_id: The user whose constitution to check.
        action_type: One of the ActionType values.
        context: Key-value pairs describing the action (e.g. data_type,
            recipient, severity_level).

    Returns:
        A compliance result with verdict, violated articles, and reasoning.
    """
    constitution = _constitutions.get(user_id)
    if constitution is None:
        # No constitution -> default allow
        return {
            "user_id": user_id,
            "action_type": action_type,
            "verdict": ComplianceVerdict.ALLOWED.value,
            "reason": "No constitution exists for this user. Action allowed by default.",
            "matched_articles": [],
            "deciding_article": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    ctx = context or {}

    # Find all matching articles
    matched: List[Dict[str, Any]] = []
    for article in constitution["articles"].values():
        if _article_matches_action(article, action_type, ctx):
            matched.append(article)

    resolution = _resolve_conflicts(matched)

    return {
        "user_id": user_id,
        "action_type": action_type,
        "verdict": resolution["verdict"],
        "reason": resolution["reason"],
        "matched_articles": [
            {"id": a["id"], "title": a["title"], "effect": a["effect"], "priority": a["priority"]}
            for a in matched
        ],
        "deciding_article": resolution["deciding_article"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def check_compliance(
    user_id: str,
    actions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Batch-check multiple actions against the user's constitution.

    Each action dict must have "action_type" and optionally "context".

    Returns a summary with per-action results and overall compliance.
    """
    results: List[Dict[str, Any]] = []
    all_allowed = True

    for action in actions:
        action_type = action.get("action_type", "")
        context = action.get("context", {})
        result = evaluate_action(user_id, action_type, context=context)
        results.append(result)
        if result["verdict"] != ComplianceVerdict.ALLOWED.value:
            all_allowed = False

    return {
        "user_id": user_id,
        "total_actions": len(actions),
        "all_compliant": all_allowed,
        "denied_count": sum(
            1 for r in results if r["verdict"] == ComplianceVerdict.DENIED.value
        ),
        "conditional_count": sum(
            1 for r in results if r["verdict"] == ComplianceVerdict.CONDITIONAL.value
        ),
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# History and profile
# ---------------------------------------------------------------------------

def get_constitution_history(user_id: str) -> List[Dict[str, Any]]:
    """Return the full amendment history for a user's constitution."""
    return deepcopy(_amendment_history.get(user_id, []))


def compute_constitution_profile(user_id: str) -> Dict[str, Any]:
    """Compute an analytical profile of the user's constitution.

    Summarizes article counts per domain, average priority, coverage of
    action types, and amendment activity.
    """
    constitution = _constitutions.get(user_id)
    if constitution is None:
        return {
            "user_id": user_id,
            "exists": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    articles = list(constitution["articles"].values())
    history = _amendment_history.get(user_id, [])

    # Per-domain stats
    domain_counts: Dict[str, int] = {d.value: 0 for d in ArticleDomain}
    domain_priorities: Dict[str, List[int]] = {d.value: [] for d in ArticleDomain}
    effect_counts: Dict[str, int] = {"deny": 0, "allow": 0, "conditional": 0}
    covered_action_types: set = set()

    for article in articles:
        domain = article["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        domain_priorities.setdefault(domain, []).append(article["priority"])
        effect_counts[article["effect"]] = effect_counts.get(article["effect"], 0) + 1
        covered_action_types.update(article.get("action_types", []))

    all_action_types = {a.value for a in ActionType}
    uncovered = all_action_types - covered_action_types

    avg_priorities: Dict[str, float] = {}
    for domain, plist in domain_priorities.items():
        if plist:
            avg_priorities[domain] = round(sum(plist) / len(plist), 1)

    return {
        "user_id": user_id,
        "exists": True,
        "name": constitution["name"],
        "version": constitution["version"],
        "total_articles": len(articles),
        "total_amendments": len(history),
        "articles_per_domain": domain_counts,
        "average_priority_per_domain": avg_priorities,
        "effect_distribution": effect_counts,
        "covered_action_types": sorted(covered_action_types),
        "uncovered_action_types": sorted(uncovered),
        "coverage_ratio": round(
            len(covered_action_types) / max(len(all_action_types), 1), 2
        ),
        "created_at": constitution["created_at"],
        "updated_at": constitution["updated_at"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def profile_to_dict(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a constitution profile to a JSON-safe dict.

    Converts any enum keys/values to their string representation.
    """
    def _convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                (k.value if isinstance(k, Enum) else k): _convert(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [_convert(item) for item in obj]
        if isinstance(obj, Enum):
            return obj.value
        return obj

    return _convert(profile)
