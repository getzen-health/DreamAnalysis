"""Tests for PersonalModel personalization status and progress."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.personal_model import PersonalModel


def test_status_reports_progress_before_activation():
    pm = PersonalModel(user_id="pytest_status_user")
    pm.total_sessions = 3
    pm.total_labeled_epochs = 3

    status = pm.status()

    assert status["personal_model_active"] is False
    assert status["personalization_progress_pct"] == 60
    assert status["activation_threshold_sessions"] == 5
    assert "3/5 corrected sessions" in status["message"]


def test_status_reports_accuracy_and_priors_after_activation():
    pm = PersonalModel(user_id="pytest_status_active")
    pm.total_sessions = 5
    pm.total_labeled_epochs = 8
    pm.head_accuracy = 0.86
    pm.feature_priors = {
        "alpha_mean": 0.12,
        "beta_mean": 0.08,
        "theta_mean": 0.05,
    }
    pm._buffer_y = [0, 1, 2, 3, 4]

    status = pm.status()

    assert status["personal_model_active"] is True
    assert status["personalization_progress_pct"] == 100
    assert status["accuracy_improvement_pct"] == 15.0
    assert status["personal_blend_weight_pct"] == 70
    assert status["feature_priors"]["alpha_mean"] == 0.12
