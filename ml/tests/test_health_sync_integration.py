"""Integration tests for Apple HealthKit and Google Fit / Health Connect modules."""

import pytest


# ── Apple HealthKit ─────────────────────────────────────────────────────────────

class TestHealthkitTypeMap:
    def test_map_has_17_types(self):
        from health.apple_health import HEALTHKIT_TYPE_MAP
        assert len(HEALTHKIT_TYPE_MAP) == 17

    def test_vital_signs_present(self):
        from health.apple_health import HEALTHKIT_TYPE_MAP
        assert "HKQuantityTypeIdentifierHeartRate" in HEALTHKIT_TYPE_MAP
        assert HEALTHKIT_TYPE_MAP["HKQuantityTypeIdentifierHeartRate"] == "heart_rate"

    def test_hrv_present(self):
        from health.apple_health import HEALTHKIT_TYPE_MAP
        assert "HKQuantityTypeIdentifierHeartRateVariabilitySDNN" in HEALTHKIT_TYPE_MAP
        assert HEALTHKIT_TYPE_MAP["HKQuantityTypeIdentifierHeartRateVariabilitySDNN"] == "hrv_sdnn"

    def test_sleep_present(self):
        from health.apple_health import HEALTHKIT_TYPE_MAP
        assert "HKCategoryTypeIdentifierSleepAnalysis" in HEALTHKIT_TYPE_MAP

    def test_mindfulness_present(self):
        from health.apple_health import HEALTHKIT_TYPE_MAP
        assert "HKCategoryTypeIdentifierMindfulSession" in HEALTHKIT_TYPE_MAP

    def test_activity_types_present(self):
        from health.apple_health import HEALTHKIT_TYPE_MAP
        for key in [
            "HKQuantityTypeIdentifierStepCount",
            "HKQuantityTypeIdentifierActiveEnergyBurned",
            "HKQuantityTypeIdentifierAppleExerciseTime",
        ]:
            assert key in HEALTHKIT_TYPE_MAP


class TestParseHealthkitPayload:
    def _make_payload(self, hk_type, samples):
        return {"type": hk_type, "samples": samples}

    def test_valid_heart_rate_payload(self):
        from health.apple_health import parse_healthkit_payload
        payload = self._make_payload(
            "HKQuantityTypeIdentifierHeartRate",
            [
                {
                    "startDate": "2024-01-01T08:00:00",
                    "endDate": "2024-01-01T08:00:10",
                    "value": 68,
                    "unit": "count/min",
                    "sourceName": "Apple Watch",
                }
            ],
        )
        result = parse_healthkit_payload(payload)
        assert result["metric"] == "heart_rate"
        assert result["healthkit_type"] == "HKQuantityTypeIdentifierHeartRate"
        assert result["count"] == 1
        sample = result["samples"][0]
        assert sample["value"] == 68
        assert sample["timestamp"] == "2024-01-01T08:00:00"
        assert sample["source"] == "Apple Watch"

    def test_empty_samples(self):
        from health.apple_health import parse_healthkit_payload
        result = parse_healthkit_payload({"type": "HKQuantityTypeIdentifierHeartRate", "samples": []})
        assert result["count"] == 0
        assert result["samples"] == []

    def test_malformed_payload_missing_type(self):
        from health.apple_health import parse_healthkit_payload
        result = parse_healthkit_payload({"samples": []})
        # Falls back to unknown type passed through unchanged
        assert result["metric"] == ""
        assert result["count"] == 0

    def test_malformed_payload_missing_samples(self):
        from health.apple_health import parse_healthkit_payload
        result = parse_healthkit_payload({"type": "HKQuantityTypeIdentifierHeartRate"})
        assert result["count"] == 0
        assert result["samples"] == []

    def test_unknown_hk_type_passed_through(self):
        from health.apple_health import parse_healthkit_payload
        result = parse_healthkit_payload({"type": "HKQuantityTypeUnknownFuture", "samples": []})
        # Unknown types are passed through as-is
        assert result["metric"] == "HKQuantityTypeUnknownFuture"
        assert result["healthkit_type"] == "HKQuantityTypeUnknownFuture"

    def test_sample_defaults_for_missing_fields(self):
        from health.apple_health import parse_healthkit_payload
        result = parse_healthkit_payload({
            "type": "HKQuantityTypeIdentifierStepCount",
            "samples": [{"value": 1000}],
        })
        sample = result["samples"][0]
        assert sample["timestamp"] is None
        assert sample["unit"] == ""
        assert sample["source"] == "Apple Health"
        assert sample["device"] == ""

    def test_multiple_samples_counted(self):
        from health.apple_health import parse_healthkit_payload
        samples = [{"value": i, "startDate": f"2024-01-01T0{i}:00:00"} for i in range(5)]
        result = parse_healthkit_payload({
            "type": "HKQuantityTypeIdentifierStepCount",
            "samples": samples,
        })
        assert result["count"] == 5
        assert len(result["samples"]) == 5


class TestHealthkitSleepStageMapping:
    def test_sleep_stage_in_bed(self):
        from health.apple_health import parse_healthkit_payload
        result = parse_healthkit_payload({
            "type": "HKCategoryTypeIdentifierSleepAnalysis",
            "samples": [{"value": 0, "startDate": "2024-01-01T22:00:00", "endDate": "2024-01-01T22:30:00"}],
        })
        assert result["samples"][0]["sleep_stage"] == "in_bed"

    def test_sleep_stage_deep(self):
        from health.apple_health import parse_healthkit_payload
        result = parse_healthkit_payload({
            "type": "HKCategoryTypeIdentifierSleepAnalysis",
            "samples": [{"value": 4, "startDate": "2024-01-01T01:00:00", "endDate": "2024-01-01T01:30:00"}],
        })
        assert result["samples"][0]["sleep_stage"] == "deep_sleep"

    def test_sleep_stage_rem(self):
        from health.apple_health import parse_healthkit_payload
        result = parse_healthkit_payload({
            "type": "HKCategoryTypeIdentifierSleepAnalysis",
            "samples": [{"value": 5, "startDate": "2024-01-01T03:00:00", "endDate": "2024-01-01T03:30:00"}],
        })
        assert result["samples"][0]["sleep_stage"] == "rem_sleep"

    def test_sleep_stage_unknown_value(self):
        from health.apple_health import parse_healthkit_payload
        result = parse_healthkit_payload({
            "type": "HKCategoryTypeIdentifierSleepAnalysis",
            "samples": [{"value": 99}],
        })
        assert result["samples"][0]["sleep_stage"] == "unknown"

    def test_non_sleep_type_has_no_sleep_stage(self):
        from health.apple_health import parse_healthkit_payload
        result = parse_healthkit_payload({
            "type": "HKQuantityTypeIdentifierHeartRate",
            "samples": [{"value": 65}],
        })
        assert "sleep_stage" not in result["samples"][0]

    def test_all_six_sleep_stages_mapped(self):
        from health.apple_health import HEALTHKIT_SLEEP_STAGES
        assert len(HEALTHKIT_SLEEP_STAGES) == 6
        for stage_value in range(6):
            assert stage_value in HEALTHKIT_SLEEP_STAGES


class TestFormatBrainDataForHealthkit:
    def test_flow_state_produces_mindful_session(self):
        from health.apple_health import format_brain_data_for_healthkit
        session = {
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-01T10:30:00",
            "flow_state": {"state": "flow", "flow_score": 0.85},
            "sleep_stage": {},
        }
        exports = format_brain_data_for_healthkit(session)
        assert len(exports) == 1
        export = exports[0]
        assert export["type"] == "HKCategoryTypeIdentifierMindfulSession"
        assert export["startDate"] == "2024-01-01T10:00:00"
        assert export["metadata"]["flow_score"] == 0.85
        assert export["metadata"]["source"] == "NeuralDreamWorkshop"

    def test_deep_flow_also_exported(self):
        from health.apple_health import format_brain_data_for_healthkit
        session = {
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-01T10:30:00",
            "flow_state": {"state": "deep_flow", "flow_score": 0.95},
            "sleep_stage": {},
        }
        exports = format_brain_data_for_healthkit(session)
        assert any(e["type"] == "HKCategoryTypeIdentifierMindfulSession" for e in exports)

    def test_non_flow_state_not_exported(self):
        from health.apple_health import format_brain_data_for_healthkit
        session = {
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-01T10:30:00",
            "flow_state": {"state": "distracted", "flow_score": 0.2},
            "sleep_stage": {},
        }
        exports = format_brain_data_for_healthkit(session)
        assert not any(e["type"] == "HKCategoryTypeIdentifierMindfulSession" for e in exports)

    def test_sleep_stage_produces_sleep_analysis(self):
        from health.apple_health import format_brain_data_for_healthkit
        session = {
            "start_time": "2024-01-01T02:00:00",
            "end_time": "2024-01-01T02:30:00",
            "flow_state": {},
            "sleep_stage": {"stage": "N3", "confidence": 0.9},
        }
        exports = format_brain_data_for_healthkit(session)
        assert len(exports) == 1
        export = exports[0]
        assert export["type"] == "HKCategoryTypeIdentifierSleepAnalysis"
        assert export["value"] == 4  # N3 → deep_sleep = 4
        assert export["metadata"]["eeg_verified"] is True
        assert export["metadata"]["confidence"] == 0.9

    def test_wake_stage_maps_to_value_2(self):
        from health.apple_health import format_brain_data_for_healthkit
        session = {
            "start_time": "2024-01-01T02:00:00",
            "end_time": "2024-01-01T02:05:00",
            "flow_state": {},
            "sleep_stage": {"stage": "Wake"},
        }
        exports = format_brain_data_for_healthkit(session)
        assert exports[0]["value"] == 2  # Wake → awake

    def test_rem_maps_to_value_5(self):
        from health.apple_health import format_brain_data_for_healthkit
        session = {
            "start_time": "2024-01-01T05:00:00",
            "end_time": "2024-01-01T05:30:00",
            "flow_state": {},
            "sleep_stage": {"stage": "REM"},
        }
        exports = format_brain_data_for_healthkit(session)
        assert exports[0]["value"] == 5

    def test_flow_and_sleep_both_exported(self):
        from health.apple_health import format_brain_data_for_healthkit
        session = {
            "start_time": "2024-01-01T05:00:00",
            "end_time": "2024-01-01T05:30:00",
            "flow_state": {"state": "flow", "flow_score": 0.8},
            "sleep_stage": {"stage": "REM"},
        }
        exports = format_brain_data_for_healthkit(session)
        types = {e["type"] for e in exports}
        assert "HKCategoryTypeIdentifierMindfulSession" in types
        assert "HKCategoryTypeIdentifierSleepAnalysis" in types

    def test_empty_session_returns_no_exports(self):
        from health.apple_health import format_brain_data_for_healthkit
        exports = format_brain_data_for_healthkit({})
        assert exports == []


# ── Google Fit ──────────────────────────────────────────────────────────────────

class TestHealthConnectTypeMap:
    def test_map_has_9_types(self):
        from health.google_fit import HEALTH_CONNECT_TYPE_MAP
        assert len(HEALTH_CONNECT_TYPE_MAP) == 9

    def test_heart_rate_present(self):
        from health.google_fit import HEALTH_CONNECT_TYPE_MAP
        assert "HeartRateRecord" in HEALTH_CONNECT_TYPE_MAP
        assert HEALTH_CONNECT_TYPE_MAP["HeartRateRecord"] == "heart_rate"

    def test_sleep_session_present(self):
        from health.google_fit import HEALTH_CONNECT_TYPE_MAP
        assert "SleepSessionRecord" in HEALTH_CONNECT_TYPE_MAP

    def test_hrv_rmssd_present(self):
        from health.google_fit import HEALTH_CONNECT_TYPE_MAP
        assert "HeartRateVariabilityRmssdRecord" in HEALTH_CONNECT_TYPE_MAP
        assert HEALTH_CONNECT_TYPE_MAP["HeartRateVariabilityRmssdRecord"] == "hrv_rmssd"

    def test_all_expected_types(self):
        from health.google_fit import HEALTH_CONNECT_TYPE_MAP
        expected = {
            "HeartRateRecord",
            "StepsRecord",
            "SleepSessionRecord",
            "OxygenSaturationRecord",
            "RespiratoryRateRecord",
            "ExerciseSessionRecord",
            "HeartRateVariabilityRmssdRecord",
            "RestingHeartRateRecord",
            "BloodGlucoseRecord",
        }
        assert set(HEALTH_CONNECT_TYPE_MAP.keys()) == expected


class TestParseGoogleFitPayload:
    def test_valid_heart_rate_payload(self):
        from health.google_fit import parse_google_fit_payload
        payload = {
            "dataTypeName": "com.google.heart_rate.bpm",
            "point": [
                {
                    "startTimeNanos": 1704067200 * 1_000_000_000,
                    "endTimeNanos": 1704067210 * 1_000_000_000,
                    "value": [{"fpVal": 72.5}],
                }
            ],
        }
        result = parse_google_fit_payload(payload)
        assert result["metric"] == "heart_rate"
        assert result["count"] == 1
        assert result["samples"][0]["value"] == 72.5

    def test_nanosecond_timestamp_conversion(self):
        from health.google_fit import parse_google_fit_payload
        # 1704067200 seconds = 2024-01-01T00:00:00 UTC
        ns = 1704067200 * 1_000_000_000
        payload = {
            "dataTypeName": "com.google.heart_rate.bpm",
            "point": [
                {
                    "startTimeNanos": ns,
                    "endTimeNanos": ns + 10_000_000_000,
                    "value": [{"fpVal": 68.0}],
                }
            ],
        }
        result = parse_google_fit_payload(payload)
        # nanoseconds should be converted to seconds
        assert abs(result["samples"][0]["timestamp"] - 1704067200) < 1

    def test_int_val_extracted(self):
        from health.google_fit import parse_google_fit_payload
        payload = {
            "dataTypeName": "com.google.step_count.delta",
            "point": [
                {
                    "startTimeNanos": 1000 * 1_000_000_000,
                    "endTimeNanos": 2000 * 1_000_000_000,
                    "value": [{"intVal": 500}],
                }
            ],
        }
        result = parse_google_fit_payload(payload)
        assert result["metric"] == "steps"
        assert result["samples"][0]["value"] == 500

    def test_empty_points(self):
        from health.google_fit import parse_google_fit_payload
        result = parse_google_fit_payload({
            "dataTypeName": "com.google.heart_rate.bpm",
            "point": [],
        })
        assert result["count"] == 0
        assert result["samples"] == []

    def test_samples_key_accepted_as_alias(self):
        from health.google_fit import parse_google_fit_payload
        payload = {
            "dataTypeName": "com.google.step_count.delta",
            "samples": [
                {
                    "startTimeNanos": 0,
                    "endTimeNanos": 0,
                    "value": [{"intVal": 100}],
                }
            ],
        }
        result = parse_google_fit_payload(payload)
        assert result["count"] == 1

    def test_unknown_type_passed_through(self):
        from health.google_fit import parse_google_fit_payload
        result = parse_google_fit_payload({"dataTypeName": "com.google.unknown.future", "point": []})
        assert result["metric"] == "com.google.unknown.future"

    def test_google_fit_type_preserved(self):
        from health.google_fit import parse_google_fit_payload
        result = parse_google_fit_payload({"dataTypeName": "com.google.heart_rate.bpm", "point": []})
        assert result["google_fit_type"] == "com.google.heart_rate.bpm"


class TestGoogleFitSleepStageMapping:
    def test_sleep_stage_in_google_fit_payload(self):
        from health.google_fit import parse_google_fit_payload
        payload = {
            "dataTypeName": "com.google.sleep.segment",
            "point": [
                {
                    "startTimeNanos": 1000 * 1_000_000_000,
                    "endTimeNanos": 2000 * 1_000_000_000,
                    "value": [{"intVal": 5}],  # deep_sleep
                }
            ],
        }
        result = parse_google_fit_payload(payload)
        assert result["samples"][0]["sleep_stage"] == "deep_sleep"

    def test_sleep_stage_rem(self):
        from health.google_fit import parse_google_fit_payload
        payload = {
            "dataTypeName": "com.google.sleep.segment",
            "point": [
                {
                    "startTimeNanos": 1000 * 1_000_000_000,
                    "endTimeNanos": 2000 * 1_000_000_000,
                    "value": [{"intVal": 6}],  # rem_sleep
                }
            ],
        }
        result = parse_google_fit_payload(payload)
        assert result["samples"][0]["sleep_stage"] == "rem_sleep"

    def test_non_sleep_type_no_sleep_stage(self):
        from health.google_fit import parse_google_fit_payload
        payload = {
            "dataTypeName": "com.google.heart_rate.bpm",
            "point": [
                {
                    "startTimeNanos": 1000 * 1_000_000_000,
                    "endTimeNanos": 2000 * 1_000_000_000,
                    "value": [{"fpVal": 65.0}],
                }
            ],
        }
        result = parse_google_fit_payload(payload)
        assert "sleep_stage" not in result["samples"][0]


class TestParseHealthConnectPayload:
    def test_valid_heart_rate_record(self):
        from health.google_fit import parse_health_connect_payload
        payload = {
            "recordType": "HeartRateRecord",
            "records": [
                {
                    "time": "2024-01-01T08:00:00",
                    "bpm": 72,
                    "dataOrigin": {"packageName": "com.fitbit"},
                }
            ],
        }
        result = parse_health_connect_payload(payload)
        assert result["metric"] == "heart_rate"
        assert result["count"] == 1
        assert result["samples"][0]["value"] == 72
        assert result["samples"][0]["source"] == "com.fitbit"

    def test_steps_record(self):
        from health.google_fit import parse_health_connect_payload
        payload = {
            "recordType": "StepsRecord",
            "records": [
                {
                    "startTime": "2024-01-01T08:00:00",
                    "endTime": "2024-01-01T09:00:00",
                    "count": 1200,
                    "dataOrigin": {"packageName": "com.google.android.apps.fitness"},
                }
            ],
        }
        result = parse_health_connect_payload(payload)
        assert result["metric"] == "steps"
        assert result["samples"][0]["value"] == 1200

    def test_sleep_session_with_stages(self):
        from health.google_fit import parse_health_connect_payload
        payload = {
            "recordType": "SleepSessionRecord",
            "records": [
                {
                    "startTime": "2024-01-01T22:00:00",
                    "endTime": "2024-01-02T06:00:00",
                    "stages": [
                        {"stage": 5, "startTime": "2024-01-01T23:00:00", "endTime": "2024-01-01T23:30:00"},
                        {"stage": 6, "startTime": "2024-01-01T23:30:00", "endTime": "2024-01-02T00:00:00"},
                    ],
                    "dataOrigin": {},
                }
            ],
        }
        result = parse_health_connect_payload(payload)
        assert result["metric"] == "sleep_analysis"
        stages = result["samples"][0]["sleep_stages"]
        assert len(stages) == 2
        assert stages[0]["stage"] == "deep_sleep"
        assert stages[1]["stage"] == "rem_sleep"

    def test_empty_records(self):
        from health.google_fit import parse_health_connect_payload
        result = parse_health_connect_payload({"recordType": "HeartRateRecord", "records": []})
        assert result["count"] == 0

    def test_record_type_preserved(self):
        from health.google_fit import parse_health_connect_payload
        result = parse_health_connect_payload({"recordType": "OxygenSaturationRecord", "records": []})
        assert result["record_type"] == "OxygenSaturationRecord"
        assert result["metric"] == "blood_oxygen"

    def test_unknown_record_type_passed_through(self):
        from health.google_fit import parse_health_connect_payload
        result = parse_health_connect_payload({"recordType": "FutureMetricRecord", "records": []})
        assert result["metric"] == "FutureMetricRecord"

    def test_default_source_when_no_data_origin(self):
        from health.google_fit import parse_health_connect_payload
        payload = {
            "recordType": "HeartRateRecord",
            "records": [{"time": "2024-01-01T08:00:00", "bpm": 70, "dataOrigin": {}}],
        }
        result = parse_health_connect_payload(payload)
        assert result["samples"][0]["source"] == "Health Connect"
