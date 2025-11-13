"""Unit tests for Instance domain model."""

import pytest

from sec_bench_sdk.domain.models.instance import Instance


class TestInstance:
    """Tests for Instance model."""

    def test_create_instance_with_required_fields(self):
        """Test creating an instance with required fields."""
        instance = Instance(
            instance_id="test.cve-2022-1234",
            repository_url="https://github.com/test/repo",
            base_commit="abc123",
            vulnerability_description="Test vulnerability",
        )

        assert instance.instance_id == "test.cve-2022-1234"
        assert instance.repository_url == "https://github.com/test/repo"
        assert instance.base_commit == "abc123"
        assert instance.vulnerability_description == "Test vulnerability"

    def test_create_instance_with_optional_fields(self):
        """Test creating an instance with optional fields."""
        instance = Instance(
            instance_id="test.cve-2022-1234",
            repository_url="https://github.com/test/repo",
            base_commit="abc123",
            vulnerability_description="Test vulnerability",
            cve_id="CVE-2022-1234",
            sanitizer_type="AddressSanitizer",
            additional_metadata={"key": "value"},
        )

        assert instance.cve_id == "CVE-2022-1234"
        assert instance.sanitizer_type == "AddressSanitizer"
        assert instance.additional_metadata == {"key": "value"}

    def test_instance_is_immutable(self):
        """Test that Instance is immutable (frozen dataclass)."""
        instance = Instance(
            instance_id="test.cve-2022-1234",
            repository_url="https://github.com/test/repo",
            base_commit="abc123",
            vulnerability_description="Test vulnerability",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            instance.instance_id = "new_id"

    def test_instance_validation_empty_id(self):
        """Test that empty instance_id raises ValueError."""
        with pytest.raises(ValueError, match="instance_id cannot be empty"):
            Instance(
                instance_id="",
                repository_url="https://github.com/test/repo",
                base_commit="abc123",
                vulnerability_description="Test vulnerability",
            )

    def test_instance_validation_empty_repository(self):
        """Test that empty repository_url raises ValueError."""
        with pytest.raises(ValueError, match="repository_url cannot be empty"):
            Instance(
                instance_id="test.cve",
                repository_url="",
                base_commit="abc123",
                vulnerability_description="Test vulnerability",
            )

    def test_instance_validation_empty_commit(self):
        """Test that empty base_commit raises ValueError."""
        with pytest.raises(ValueError, match="base_commit cannot be empty"):
            Instance(
                instance_id="test.cve",
                repository_url="https://github.com/test/repo",
                base_commit="",
                vulnerability_description="Test vulnerability",
            )
