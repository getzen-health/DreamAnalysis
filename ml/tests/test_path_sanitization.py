"""Tests for path traversal prevention in ID sanitization."""
import pytest
from fastapi import HTTPException

from api.routes._shared import sanitize_id


class TestSanitizeId:
    def test_valid_ids(self):
        assert sanitize_id("user123") == "user123"
        assert sanitize_id("test-user_42") == "test-user_42"

    def test_path_traversal_rejected(self):
        with pytest.raises(HTTPException):
            sanitize_id("../../etc")

    def test_slash_rejected(self):
        with pytest.raises(HTTPException):
            sanitize_id("user/evil")

    def test_backslash_rejected(self):
        with pytest.raises(HTTPException):
            sanitize_id("user\\evil")

    def test_dots_rejected(self):
        with pytest.raises(HTTPException):
            sanitize_id("..user")

    def test_empty_rejected(self):
        with pytest.raises(HTTPException):
            sanitize_id("")

    def test_too_long_rejected(self):
        with pytest.raises(HTTPException):
            sanitize_id("a" * 129)

    def test_custom_field_name_in_detail(self):
        with pytest.raises(HTTPException) as exc_info:
            sanitize_id("bad/id", "user_id")
        assert "user_id" in exc_info.value.detail

    def test_status_code_422(self):
        with pytest.raises(HTTPException) as exc_info:
            sanitize_id("../attack")
        assert exc_info.value.status_code == 422
