"""Apple HealthKit Integration Module.

Maps Apple HealthKit data types to our internal schema and provides
utilities for importing/exporting health data.

HealthKit Categories Supported:
- Heart Rate (HKQuantityTypeIdentifierHeartRate)
- Heart Rate Variability (HKQuantityTypeIdentifierHeartRateVariabilitySDNN)
- Sleep Analysis (HKCategoryTypeIdentifierSleepAnalysis)
- Steps (HKQuantityTypeIdentifierStepCount)
- Active Energy (HKQuantityTypeIdentifierActiveEnergyBurned)
- Blood Oxygen (HKQuantityTypeIdentifierOxygenSaturation)
- Respiratory Rate (HKQuantityTypeIdentifierRespiratoryRate)
- Mindful Minutes (HKCategoryTypeIdentifierMindfulSession)
- Resting Heart Rate (HKQuantityTypeIdentifierRestingHeartRate)
- Walking Heart Rate Average (HKQuantityTypeIdentifierWalkingHeartRateAverage)

The actual HealthKit SDK calls happen on the iOS client.
This module handles the server-side data processing.
"""

# HealthKit type identifiers → our internal field names
HEALTHKIT_TYPE_MAP = {
    # Vital Signs
    "HKQuantityTypeIdentifierHeartRate": "heart_rate",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv_sdnn",
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_heart_rate",
    "HKQuantityTypeIdentifierWalkingHeartRateAverage": "walking_heart_rate",
    "HKQuantityTypeIdentifierOxygenSaturation": "blood_oxygen",
    "HKQuantityTypeIdentifierRespiratoryRate": "respiratory_rate",
    "HKQuantityTypeIdentifierBodyTemperature": "body_temperature",

    # Activity
    "HKQuantityTypeIdentifierStepCount": "steps",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_calories",
    "HKQuantityTypeIdentifierBasalEnergyBurned": "basal_calories",
    "HKQuantityTypeIdentifierDistanceWalkingRunning": "distance_walking",
    "HKQuantityTypeIdentifierAppleExerciseTime": "exercise_minutes",
    "HKQuantityTypeIdentifierAppleStandTime": "stand_minutes",

    # Sleep
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep_analysis",

    # Mindfulness
    "HKCategoryTypeIdentifierMindfulSession": "mindful_minutes",

    # Noise
    "HKQuantityTypeIdentifierEnvironmentalAudioExposure": "noise_exposure",

    # Metabolic
    "HKQuantityTypeIdentifierBloodGlucose": "blood_glucose",
}

# HealthKit sleep stage values → our sleep stage names
HEALTHKIT_SLEEP_STAGES = {
    0: "in_bed",           # HKCategoryValueSleepAnalysisInBed
    1: "asleep",           # HKCategoryValueSleepAnalysisAsleep (unspecified)
    2: "awake",            # HKCategoryValueSleepAnalysisAwake
    3: "core_sleep",       # HKCategoryValueSleepAnalysisAsleepCore (N1+N2)
    4: "deep_sleep",       # HKCategoryValueSleepAnalysisAsleepDeep (N3)
    5: "rem_sleep",        # HKCategoryValueSleepAnalysisAsleepREM
}

# Brain data types we can export TO Apple Health
EXPORT_TYPES = {
    "mindful_minutes": {
        "healthkit_type": "HKCategoryTypeIdentifierMindfulSession",
        "description": "Brain-detected meditation/flow sessions",
    },
    "sleep_analysis": {
        "healthkit_type": "HKCategoryTypeIdentifierSleepAnalysis",
        "description": "EEG-verified sleep stages",
    },
}


def parse_healthkit_payload(payload: dict) -> dict:
    """Parse raw HealthKit export data into our internal format.

    Args:
        payload: Dict with 'type' (HealthKit identifier) and 'samples' list.
                 Each sample has 'startDate', 'endDate', 'value'.

    Returns:
        Normalized dict with internal field names and parsed values.
    """
    hk_type = payload.get("type", "")
    internal_name = HEALTHKIT_TYPE_MAP.get(hk_type, hk_type)
    samples = payload.get("samples", [])

    parsed_samples = []
    for sample in samples:
        parsed = {
            "timestamp": sample.get("startDate"),
            "end_timestamp": sample.get("endDate"),
            "value": sample.get("value"),
            "unit": sample.get("unit", ""),
            "source": sample.get("sourceName", "Apple Health"),
            "device": sample.get("device", ""),
        }

        # Parse sleep stages
        if internal_name == "sleep_analysis" and isinstance(parsed["value"], (int, float)):
            parsed["sleep_stage"] = HEALTHKIT_SLEEP_STAGES.get(int(parsed["value"]), "unknown")

        parsed_samples.append(parsed)

    return {
        "metric": internal_name,
        "healthkit_type": hk_type,
        "samples": parsed_samples,
        "count": len(parsed_samples),
    }


def format_brain_data_for_healthkit(brain_session: dict) -> list:
    """Format brain state data for export to Apple HealthKit.

    Args:
        brain_session: Session data with brain state analysis.

    Returns:
        List of HealthKit-compatible sample dicts.
    """
    exports = []

    # Export flow/meditation sessions as Mindful Minutes
    flow_data = brain_session.get("flow_state", {})
    if flow_data.get("state") in ("flow", "deep_flow"):
        exports.append({
            "type": "HKCategoryTypeIdentifierMindfulSession",
            "startDate": brain_session.get("start_time"),
            "endDate": brain_session.get("end_time"),
            "value": 0,  # HKCategoryValueNotApplicable
            "metadata": {
                "flow_score": flow_data.get("flow_score"),
                "flow_state": flow_data.get("state"),
                "source": "NeuralDreamWorkshop",
            },
        })

    # Export EEG sleep staging
    sleep_data = brain_session.get("sleep_stage", {})
    if sleep_data.get("stage"):
        stage_map = {
            "Wake": 2, "N1": 3, "N2": 3, "N3": 4, "REM": 5,
        }
        hk_value = stage_map.get(sleep_data["stage"], 1)
        exports.append({
            "type": "HKCategoryTypeIdentifierSleepAnalysis",
            "startDate": brain_session.get("start_time"),
            "endDate": brain_session.get("end_time"),
            "value": hk_value,
            "metadata": {
                "eeg_verified": True,
                "confidence": sleep_data.get("confidence"),
                "source": "NeuralDreamWorkshop",
            },
        })

    return exports
