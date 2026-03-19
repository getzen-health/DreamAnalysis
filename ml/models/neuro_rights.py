"""Neuro-rights governance framework.

Implements the five fundamental neuro-rights proposed by Yuste & Goering (2017):
  1. Mental privacy   -- right to keep neural data confidential
  2. Cognitive liberty -- right to freely alter one's own mental states
  3. Mental integrity  -- right to be protected from unauthorized alteration
  4. Psychological continuity -- right to preserve personal identity
  5. Fair access       -- right to equitable access to neurotechnology

Provides:
  - Rights compliance auditing against configurable data flows
  - Data sovereignty inventory (what exists, who accesses, when shared)
  - Immutable consent ledger with timestamps
  - Right-to-delete (GDPR Art 17) impact analysis
  - Right-to-explanation for ML predictions
  - Data minimization assessment

GitHub issue: #443
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class NeuroRight(str, Enum):
    """The five fundamental neuro-rights (Yuste & Goering, 2017)."""
    MENTAL_PRIVACY = "mental_privacy"
    COGNITIVE_LIBERTY = "cognitive_liberty"
    MENTAL_INTEGRITY = "mental_integrity"
    PSYCHOLOGICAL_CONTINUITY = "psychological_continuity"
    FAIR_ACCESS = "fair_access"


NEURO_RIGHTS_DESCRIPTIONS: Dict[str, str] = {
    NeuroRight.MENTAL_PRIVACY: (
        "The right to keep neural data confidential and prevent unauthorized "
        "access to brain information."
    ),
    NeuroRight.COGNITIVE_LIBERTY: (
        "The right to freely alter one's own mental states using neurotechnology "
        "without coercion."
    ),
    NeuroRight.MENTAL_INTEGRITY: (
        "The right to be protected from unauthorized manipulation or alteration "
        "of neural activity."
    ),
    NeuroRight.PSYCHOLOGICAL_CONTINUITY: (
        "The right to preserve personal identity and the continuity of one's "
        "mental life against disruptive neurotechnology."
    ),
    NeuroRight.FAIR_ACCESS: (
        "The right to equitable access to neurotechnology and its cognitive "
        "enhancements, preventing a neuro-divide."
    ),
}

# Neural data categories tracked by the system
NEURAL_DATA_TYPES = frozenset({
    "eeg_raw",
    "eeg_features",
    "emotion_predictions",
    "sleep_staging",
    "dream_data",
    "cognitive_metrics",
    "neurofeedback_protocols",
    "brain_biomarkers",
    "voice_biomarkers",
    "mood_data",
})

# Valid consent actions
CONSENT_ACTIONS = frozenset({"grant", "revoke"})


# ---------------------------------------------------------------------------
# In-memory stores (per-user)
# ---------------------------------------------------------------------------

# user_id -> list of data inventory records
_data_inventory: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

# Immutable consent ledger: list of all consent events
_consent_ledger: List[Dict[str, Any]] = []

# user_id -> { data_type: bool } active consent state
_active_consents: Dict[str, Dict[str, bool]] = defaultdict(dict)

# user_id -> list of data flow records (who accessed what, when)
_data_flows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)


def _reset_stores() -> None:
    """Reset all in-memory stores. For testing only."""
    _data_inventory.clear()
    _consent_ledger.clear()
    _active_consents.clear()
    _data_flows.clear()


# ---------------------------------------------------------------------------
# Data inventory management
# ---------------------------------------------------------------------------

def register_data_record(
    user_id: str,
    data_type: str,
    *,
    source: str = "system",
    purpose: str = "analysis",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register a neural data record in the inventory.

    Returns the created inventory entry.
    """
    if data_type not in NEURAL_DATA_TYPES:
        raise ValueError(
            f"Unknown data type: {data_type}. "
            f"Valid types: {sorted(NEURAL_DATA_TYPES)}"
        )

    record = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "data_type": data_type,
        "source": source,
        "purpose": purpose,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    _data_inventory[user_id].append(record)
    return record


def register_data_flow(
    user_id: str,
    data_type: str,
    *,
    accessor: str,
    purpose: str = "analysis",
) -> Dict[str, Any]:
    """Record an access event to a user's neural data."""
    flow = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "data_type": data_type,
        "accessor": accessor,
        "purpose": purpose,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _data_flows[user_id].append(flow)
    return flow


# ---------------------------------------------------------------------------
# Consent ledger
# ---------------------------------------------------------------------------

def log_consent(
    user_id: str,
    data_type: str,
    action: str,
    *,
    purpose: str = "analysis",
    granted_to: str = "system",
) -> Dict[str, Any]:
    """Log a consent grant or revocation to the immutable ledger.

    Args:
        user_id: The user granting/revoking consent.
        data_type: The neural data type this consent applies to.
        action: Either "grant" or "revoke".
        purpose: Why consent is being granted/revoked.
        granted_to: The entity receiving (or losing) consent.

    Returns:
        The consent ledger entry.
    """
    if action not in CONSENT_ACTIONS:
        raise ValueError(f"Invalid action: {action}. Must be one of {sorted(CONSENT_ACTIONS)}")
    if data_type not in NEURAL_DATA_TYPES:
        raise ValueError(
            f"Unknown data type: {data_type}. "
            f"Valid types: {sorted(NEURAL_DATA_TYPES)}"
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "data_type": data_type,
        "action": action,
        "purpose": purpose,
        "granted_to": granted_to,
        "timestamp": timestamp,
        # Hash for immutability verification
        "hash": hashlib.sha256(
            f"{user_id}:{data_type}:{action}:{timestamp}".encode()
        ).hexdigest(),
    }

    _consent_ledger.append(entry)

    # Update active consent state
    _active_consents[user_id][data_type] = (action == "grant")

    log.info(
        "Consent %s: user=%s data_type=%s granted_to=%s",
        action, user_id, data_type, granted_to,
    )
    return entry


def get_consent_ledger(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return consent ledger entries, optionally filtered by user."""
    if user_id is None:
        return list(_consent_ledger)
    return [e for e in _consent_ledger if e["user_id"] == user_id]


def check_consent(user_id: str, data_type: str) -> bool:
    """Check if a user has active consent for a given data type."""
    return _active_consents.get(user_id, {}).get(data_type, False)


# ---------------------------------------------------------------------------
# Core governance functions
# ---------------------------------------------------------------------------

def audit_rights_compliance(
    user_id: str,
    data_flows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Audit all data flows for potential neuro-rights violations.

    Checks each of the five neuro-rights against the user's data flows,
    consent state, and data inventory.

    Returns a compliance report with per-right status and any violations found.
    """
    flows = data_flows if data_flows is not None else _data_flows.get(user_id, [])
    inventory = _data_inventory.get(user_id, [])
    consents = _active_consents.get(user_id, {})

    violations: List[Dict[str, Any]] = []
    right_statuses: Dict[str, Dict[str, Any]] = {}

    # 1. Mental privacy: check for data accessed without consent
    unconsented_flows = []
    for flow in flows:
        dt = flow.get("data_type", "")
        if not consents.get(dt, False):
            unconsented_flows.append(flow)

    mental_privacy_ok = len(unconsented_flows) == 0
    right_statuses[NeuroRight.MENTAL_PRIVACY] = {
        "compliant": mental_privacy_ok,
        "description": NEURO_RIGHTS_DESCRIPTIONS[NeuroRight.MENTAL_PRIVACY],
        "issues": (
            []
            if mental_privacy_ok
            else [
                {
                    "type": "unconsented_access",
                    "data_type": f["data_type"],
                    "accessor": f.get("accessor", "unknown"),
                    "timestamp": f.get("timestamp"),
                }
                for f in unconsented_flows
            ]
        ),
    }
    if not mental_privacy_ok:
        violations.append({
            "right": NeuroRight.MENTAL_PRIVACY,
            "count": len(unconsented_flows),
            "severity": "high",
        })

    # 2. Cognitive liberty: check that neurofeedback protocols
    #    are user-initiated (not externally imposed)
    nf_flows = [f for f in flows if f.get("data_type") == "neurofeedback_protocols"]
    external_nf = [f for f in nf_flows if f.get("accessor") != user_id and f.get("accessor") != "system"]
    cognitive_liberty_ok = len(external_nf) == 0
    right_statuses[NeuroRight.COGNITIVE_LIBERTY] = {
        "compliant": cognitive_liberty_ok,
        "description": NEURO_RIGHTS_DESCRIPTIONS[NeuroRight.COGNITIVE_LIBERTY],
        "issues": (
            []
            if cognitive_liberty_ok
            else [
                {
                    "type": "external_neurofeedback",
                    "accessor": f.get("accessor"),
                    "timestamp": f.get("timestamp"),
                }
                for f in external_nf
            ]
        ),
    }
    if not cognitive_liberty_ok:
        violations.append({
            "right": NeuroRight.COGNITIVE_LIBERTY,
            "count": len(external_nf),
            "severity": "critical",
        })

    # 3. Mental integrity: check for write/modification flows
    #    (any flow that is not read-only to brain data)
    modification_flows = [
        f for f in flows
        if f.get("purpose") in ("modification", "stimulation", "alteration")
        and not consents.get(f.get("data_type", ""), False)
    ]
    mental_integrity_ok = len(modification_flows) == 0
    right_statuses[NeuroRight.MENTAL_INTEGRITY] = {
        "compliant": mental_integrity_ok,
        "description": NEURO_RIGHTS_DESCRIPTIONS[NeuroRight.MENTAL_INTEGRITY],
        "issues": (
            []
            if mental_integrity_ok
            else [
                {
                    "type": "unauthorized_modification",
                    "data_type": f.get("data_type"),
                    "accessor": f.get("accessor"),
                    "timestamp": f.get("timestamp"),
                }
                for f in modification_flows
            ]
        ),
    }
    if not mental_integrity_ok:
        violations.append({
            "right": NeuroRight.MENTAL_INTEGRITY,
            "count": len(modification_flows),
            "severity": "critical",
        })

    # 4. Psychological continuity: check if emotion/cognitive models
    #    are altering baseline identity markers without consent
    identity_types = {"emotion_predictions", "cognitive_metrics", "brain_biomarkers"}
    identity_flows = [
        f for f in flows
        if f.get("data_type") in identity_types
        and f.get("purpose") in ("modification", "alteration")
    ]
    psych_continuity_ok = len(identity_flows) == 0
    right_statuses[NeuroRight.PSYCHOLOGICAL_CONTINUITY] = {
        "compliant": psych_continuity_ok,
        "description": NEURO_RIGHTS_DESCRIPTIONS[NeuroRight.PSYCHOLOGICAL_CONTINUITY],
        "issues": (
            []
            if psych_continuity_ok
            else [
                {
                    "type": "identity_alteration",
                    "data_type": f.get("data_type"),
                    "timestamp": f.get("timestamp"),
                }
                for f in identity_flows
            ]
        ),
    }
    if not psych_continuity_ok:
        violations.append({
            "right": NeuroRight.PSYCHOLOGICAL_CONTINUITY,
            "count": len(identity_flows),
            "severity": "high",
        })

    # 5. Fair access: check that all data types in inventory
    #    are accessible to the user (no locked-out data)
    user_data_types = {r["data_type"] for r in inventory}
    locked_types = [
        dt for dt in user_data_types
        if not consents.get(dt, True)  # default True: if no consent record, assume accessible
    ]
    fair_access_ok = len(locked_types) == 0
    right_statuses[NeuroRight.FAIR_ACCESS] = {
        "compliant": fair_access_ok,
        "description": NEURO_RIGHTS_DESCRIPTIONS[NeuroRight.FAIR_ACCESS],
        "issues": (
            []
            if fair_access_ok
            else [{"type": "access_denied", "data_type": dt} for dt in locked_types]
        ),
    }
    if not fair_access_ok:
        violations.append({
            "right": NeuroRight.FAIR_ACCESS,
            "count": len(locked_types),
            "severity": "medium",
        })

    overall_compliant = all(
        rs["compliant"] for rs in right_statuses.values()
    )

    return {
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_compliant": overall_compliant,
        "rights": {k.value if isinstance(k, NeuroRight) else k: v for k, v in right_statuses.items()},
        "violations": violations,
        "total_violations": len(violations),
        "data_flows_audited": len(flows),
        "inventory_records": len(inventory),
    }


def compute_data_inventory(user_id: str) -> Dict[str, Any]:
    """Compute a data sovereignty dashboard for a user.

    Returns what neural data exists, who has accessed it, and consent status.
    """
    inventory = _data_inventory.get(user_id, [])
    flows = _data_flows.get(user_id, [])
    consents = _active_consents.get(user_id, {})

    # Group inventory by data type
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in inventory:
        by_type[record["data_type"]].append(record)

    # Group flows by data type
    flows_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for flow in flows:
        flows_by_type[flow["data_type"]].append(flow)

    # Build per-type summary
    data_types_summary: List[Dict[str, Any]] = []
    for dt in sorted(NEURAL_DATA_TYPES):
        records = by_type.get(dt, [])
        dt_flows = flows_by_type.get(dt, [])
        accessors = list({f["accessor"] for f in dt_flows})
        data_types_summary.append({
            "data_type": dt,
            "record_count": len(records),
            "access_count": len(dt_flows),
            "accessors": accessors,
            "consent_granted": consents.get(dt, False),
            "last_accessed": (
                max((f["timestamp"] for f in dt_flows), default=None)
            ),
        })

    return {
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_records": len(inventory),
        "total_data_types": len(by_type),
        "total_access_events": len(flows),
        "consent_summary": {
            "granted": [dt for dt, v in consents.items() if v],
            "revoked": [dt for dt, v in consents.items() if not v],
        },
        "data_types": data_types_summary,
    }


def compute_deletion_impact(user_id: str) -> Dict[str, Any]:
    """Compute what data would be affected by a GDPR Art 17 deletion request.

    Returns a breakdown of records, flows, and consent entries that would
    be removed, plus downstream ML models that would lose training data.
    """
    inventory = _data_inventory.get(user_id, [])
    flows = _data_flows.get(user_id, [])
    user_consents = [e for e in _consent_ledger if e["user_id"] == user_id]

    # Identify data types with records
    affected_types = list({r["data_type"] for r in inventory})

    # Identify ML models that use these data types
    model_impact: Dict[str, List[str]] = {
        "eeg_raw": ["emotion_classifier", "sleep_staging", "dream_detector"],
        "eeg_features": ["emotion_classifier", "cognitive_load", "attention"],
        "emotion_predictions": ["mood_forecaster", "emotion_trajectory"],
        "sleep_staging": ["sleep_quality_predictor", "dream_detector"],
        "dream_data": ["dream_detector", "lucid_dream_detector"],
        "cognitive_metrics": ["cognitive_load", "attention_classifier"],
        "neurofeedback_protocols": ["neurofeedback_engine"],
        "brain_biomarkers": ["brain_age_estimator", "brain_health_score"],
        "voice_biomarkers": ["voice_emotion_model", "voice_mental_health"],
        "mood_data": ["mood_forecaster"],
    }

    affected_models: List[str] = []
    for dt in affected_types:
        affected_models.extend(model_impact.get(dt, []))
    affected_models = sorted(set(affected_models))

    return {
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "deletion_summary": {
            "data_records": len(inventory),
            "access_flow_records": len(flows),
            "consent_ledger_entries": len(user_consents),
            "affected_data_types": sorted(affected_types),
            "affected_ml_models": affected_models,
        },
        "gdpr_article": "Article 17 - Right to Erasure",
        "warning": (
            "Deletion is irreversible. All records, access logs, and "
            "consent history for this user will be permanently removed."
        ),
        "estimated_records_total": len(inventory) + len(flows) + len(user_consents),
    }


def generate_explanation(
    prediction: Dict[str, Any],
    *,
    model_name: str = "unknown",
    input_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a plain-language explanation for an ML prediction.

    Implements the right-to-explanation: for any ML prediction, provide
    a human-readable description of what inputs were used and how the
    prediction was derived.

    Args:
        prediction: The ML model output dict.
        model_name: Which model produced this prediction.
        input_summary: Optional description of what input data was used.

    Returns:
        An explanation dict with plain-language text.
    """
    # Extract key prediction fields
    label = prediction.get("emotion") or prediction.get("label") or prediction.get("stage")
    confidence = prediction.get("confidence") or prediction.get("probability")
    probabilities = prediction.get("probabilities", {})
    valence = prediction.get("valence")
    arousal = prediction.get("arousal")

    # Build plain-language explanation
    explanation_parts: List[str] = []

    explanation_parts.append(
        f"This prediction was made by the '{model_name}' model."
    )

    if input_summary:
        explanation_parts.append(f"Input data: {input_summary}")
    else:
        explanation_parts.append(
            "Input data: neural signals (EEG or derived features) "
            "collected from the user's brain-computer interface session."
        )

    if label:
        conf_pct = f"{confidence * 100:.1f}%" if confidence else "unknown"
        explanation_parts.append(
            f"The model predicted '{label}' with {conf_pct} confidence."
        )

    if probabilities:
        top_3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        probs_text = ", ".join(f"{k}: {v * 100:.1f}%" for k, v in top_3)
        explanation_parts.append(f"Top predictions: {probs_text}.")

    if valence is not None:
        direction = "positive" if valence > 0 else "negative" if valence < 0 else "neutral"
        explanation_parts.append(
            f"Emotional valence: {valence:.2f} ({direction}). "
            "This reflects the positive-negative dimension of the detected emotion."
        )

    if arousal is not None:
        level = "high" if arousal > 0.6 else "low" if arousal < 0.4 else "moderate"
        explanation_parts.append(
            f"Emotional arousal: {arousal:.2f} ({level}). "
            "This reflects the intensity or activation level of the detected emotion."
        )

    # Key factors (generic for BCI models)
    key_factors = []
    if "frontal_asymmetry" in prediction:
        faa = prediction["frontal_asymmetry"]
        direction = "left-dominant (approach)" if faa > 0 else "right-dominant (withdrawal)"
        key_factors.append(f"Frontal alpha asymmetry: {faa:.3f} ({direction})")
    if "stress_index" in prediction:
        key_factors.append(f"Stress index: {prediction['stress_index']:.2f}")
    if "focus_index" in prediction:
        key_factors.append(f"Focus index: {prediction['focus_index']:.2f}")

    return {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "explanation": " ".join(explanation_parts),
        "key_factors": key_factors,
        "prediction_summary": {
            "label": label,
            "confidence": confidence,
        },
        "data_rights_notice": (
            "You have the right to understand how predictions about your "
            "neural data are made. This explanation is provided under the "
            "right-to-explanation principle of neuro-rights governance."
        ),
    }


def check_data_minimization(
    user_id: str,
    stated_purposes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assess whether more neural data is collected than necessary.

    Compares the data types in inventory against the stated purposes.
    Flags any data type that has no clear purpose or is collected
    beyond what is needed.

    Args:
        user_id: The user to audit.
        stated_purposes: Optional mapping of data_type -> stated purpose.
            If None, uses the purposes recorded in the inventory.

    Returns:
        A minimization report with flagged over-collection.
    """
    inventory = _data_inventory.get(user_id, [])

    # Collect purposes from inventory if not provided
    if stated_purposes is None:
        stated_purposes = {}
        for record in inventory:
            dt = record["data_type"]
            if dt not in stated_purposes:
                stated_purposes[dt] = record.get("purpose", "unspecified")

    # Data types that are essential vs optional for each purpose
    purpose_requirements: Dict[str, frozenset] = {
        "emotion_analysis": frozenset({"eeg_raw", "eeg_features", "emotion_predictions"}),
        "sleep_tracking": frozenset({"eeg_raw", "eeg_features", "sleep_staging"}),
        "cognitive_assessment": frozenset({"eeg_raw", "eeg_features", "cognitive_metrics"}),
        "neurofeedback": frozenset({"eeg_raw", "eeg_features", "neurofeedback_protocols"}),
        "analysis": frozenset({"eeg_raw", "eeg_features"}),
    }

    collected_types = {r["data_type"] for r in inventory}
    flagged: List[Dict[str, Any]] = []

    for dt in collected_types:
        purpose = stated_purposes.get(dt, "unspecified")
        required = purpose_requirements.get(purpose, frozenset())

        if required and dt not in required:
            flagged.append({
                "data_type": dt,
                "stated_purpose": purpose,
                "assessment": "potentially_unnecessary",
                "reason": (
                    f"Data type '{dt}' is not required for the stated "
                    f"purpose '{purpose}'."
                ),
            })

    minimization_score = 1.0 - (len(flagged) / max(len(collected_types), 1))

    return {
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_data_types_collected": len(collected_types),
        "flagged_unnecessary": len(flagged),
        "minimization_score": round(minimization_score, 2),
        "flags": flagged,
        "recommendation": (
            "Data collection appears proportionate to stated purposes."
            if not flagged
            else (
                f"{len(flagged)} data type(s) may be collected beyond what is "
                "necessary. Consider removing or justifying their collection."
            )
        ),
    }


def compute_governance_report(user_id: str) -> Dict[str, Any]:
    """Produce a comprehensive neuro-rights governance report.

    Combines rights audit, data inventory, minimization check, and
    consent summary into a single report.
    """
    audit = audit_rights_compliance(user_id)
    inventory = compute_data_inventory(user_id)
    minimization = check_data_minimization(user_id)
    deletion = compute_deletion_impact(user_id)
    consents = get_consent_ledger(user_id)

    return {
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_compliant": audit["overall_compliant"],
        "rights_audit": audit,
        "data_inventory": inventory,
        "data_minimization": minimization,
        "deletion_impact": deletion,
        "consent_history": consents,
        "neuro_rights_reference": {
            "framework": "Yuste & Goering (2017)",
            "rights": {r.value: NEURO_RIGHTS_DESCRIPTIONS[r] for r in NeuroRight},
        },
    }


def report_to_dict(report: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a governance report to a JSON-safe dict.

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

    return _convert(report)
