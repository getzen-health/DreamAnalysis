"""Google Fit / Health Connect Integration Module.

Maps Google Fit data types to our internal schema. Supports both
legacy Google Fit API and the newer Health Connect (Android 14+).

Google Fit Data Types Supported:
- com.google.heart_rate.bpm
- com.google.heart_minutes
- com.google.step_count.delta
- com.google.calories.expended
- com.google.blood_oxygen.saturation (Health Connect)
- com.google.sleep.segment
- com.google.activity.segment
- com.google.heart_rate.variability (Health Connect)

The actual Google Fit SDK calls happen on the Android client.
This module handles server-side data processing.
"""

# Google Fit data type → our internal field names
GOOGLE_FIT_TYPE_MAP = {
    # Vital Signs
    "com.google.heart_rate.bpm": "heart_rate",
    "com.google.heart_rate.variability": "hrv_sdnn",
    "com.google.blood_oxygen.saturation": "blood_oxygen",
    "com.google.respiratory_rate": "respiratory_rate",
    "com.google.body.temperature": "body_temperature",

    # Activity
    "com.google.step_count.delta": "steps",
    "com.google.calories.expended": "active_calories",
    "com.google.distance.delta": "distance_walking",
    "com.google.active_minutes": "exercise_minutes",
    "com.google.heart_minutes": "heart_points",

    # Sleep
    "com.google.sleep.segment": "sleep_analysis",

    # Activity segments
    "com.google.activity.segment": "activity_segment",
}

# Google Fit sleep segment types
GOOGLE_FIT_SLEEP_STAGES = {
    0: "unspecified",
    1: "awake",
    2: "sleeping",      # generic
    3: "out_of_bed",
    4: "light_sleep",   # N1+N2
    5: "deep_sleep",    # N3
    6: "rem_sleep",     # REM
}

# Health Connect (Android 14+) data types
HEALTH_CONNECT_TYPE_MAP = {
    "HeartRateRecord": "heart_rate",
    "StepsRecord": "steps",
    "SleepSessionRecord": "sleep_analysis",
    "OxygenSaturationRecord": "blood_oxygen",
    "RespiratoryRateRecord": "respiratory_rate",
    "ExerciseSessionRecord": "exercise_minutes",
    "HeartRateVariabilityRmssdRecord": "hrv_rmssd",
    "RestingHeartRateRecord": "resting_heart_rate",
    "BloodGlucoseRecord": "blood_glucose",
}


def parse_google_fit_payload(payload: dict) -> dict:
    """Parse Google Fit data into our internal format.

    Args:
        payload: Dict with 'dataTypeName' and 'point' list.

    Returns:
        Normalized dict with internal field names.
    """
    data_type = payload.get("dataTypeName", "")
    internal_name = GOOGLE_FIT_TYPE_MAP.get(data_type, data_type)
    points = payload.get("point", payload.get("samples", []))

    parsed_samples = []
    for point in points:
        # Google Fit uses nanoseconds since epoch
        start_ns = int(point.get("startTimeNanos", 0))
        end_ns = int(point.get("endTimeNanos", 0))

        # Extract value from nested structure
        values = point.get("value", [])
        if isinstance(values, list) and values:
            value = values[0].get("fpVal", values[0].get("intVal", 0))
        elif isinstance(values, (int, float)):
            value = values
        else:
            value = point.get("value", 0)

        parsed = {
            "timestamp": start_ns / 1e9 if start_ns > 1e15 else start_ns,
            "end_timestamp": end_ns / 1e9 if end_ns > 1e15 else end_ns,
            "value": value,
            "source": point.get("originDataSourceId", "Google Fit"),
        }

        # Parse sleep stages
        if internal_name == "sleep_analysis" and isinstance(value, (int, float)):
            parsed["sleep_stage"] = GOOGLE_FIT_SLEEP_STAGES.get(int(value), "unknown")

        parsed_samples.append(parsed)

    return {
        "metric": internal_name,
        "google_fit_type": data_type,
        "samples": parsed_samples,
        "count": len(parsed_samples),
    }


def parse_health_connect_payload(payload: dict) -> dict:
    """Parse Health Connect (Android 14+) data into our internal format."""
    record_type = payload.get("recordType", "")
    internal_name = HEALTH_CONNECT_TYPE_MAP.get(record_type, record_type)
    records = payload.get("records", [])

    parsed_samples = []
    for record in records:
        parsed = {
            "timestamp": record.get("time", record.get("startTime")),
            "end_timestamp": record.get("endTime"),
            "value": record.get("bpm", record.get("count", record.get("percentage", 0))),
            "source": record.get("dataOrigin", {}).get("packageName", "Health Connect"),
        }

        if internal_name == "sleep_analysis":
            stages = record.get("stages", [])
            parsed["sleep_stages"] = [
                {"stage": GOOGLE_FIT_SLEEP_STAGES.get(s.get("stage", 0), "unknown"),
                 "start": s.get("startTime"), "end": s.get("endTime")}
                for s in stages
            ]

        parsed_samples.append(parsed)

    return {
        "metric": internal_name,
        "record_type": record_type,
        "samples": parsed_samples,
        "count": len(parsed_samples),
    }
