"""Unit tests for configuration module."""

from __future__ import annotations

import pytest

from quartermaster_code_runner.config import (
    SUPPORTED_IMAGES,
    Settings,
    get_image_name,
    get_short_image_name,
)


class TestImageNaming:
    """Tests for image name conversion functions."""

    def test_get_image_name_short(self) -> None:
        assert get_image_name("python") == "code-runner-python"

    def test_get_image_name_already_full(self) -> None:
        assert get_image_name("code-runner-python") == "code-runner-python"

    def test_get_image_name_prebuilt(self) -> None:
        assert get_image_name("prebuilt-myimg") == "prebuilt-myimg"

    def test_get_short_image_name_full(self) -> None:
        assert get_short_image_name("code-runner-python") == "python"

    def test_get_short_image_name_already_short(self) -> None:
        assert get_short_image_name("python") == "python"

    def test_get_short_image_name_prebuilt(self) -> None:
        # Prebuilt names are returned as-is (no runtime prefix)
        assert get_short_image_name("prebuilt-myimg") == "prebuilt-myimg"


class TestSupportedImages:
    """Tests for supported image list."""

    def test_has_python(self) -> None:
        assert "python" in SUPPORTED_IMAGES

    def test_has_node(self) -> None:
        assert "node" in SUPPORTED_IMAGES

    def test_has_go(self) -> None:
        assert "go" in SUPPORTED_IMAGES

    def test_has_rust(self) -> None:
        assert "rust" in SUPPORTED_IMAGES

    def test_has_deno(self) -> None:
        assert "deno" in SUPPORTED_IMAGES

    def test_has_bun(self) -> None:
        assert "bun" in SUPPORTED_IMAGES

    def test_six_supported(self) -> None:
        assert len(SUPPORTED_IMAGES) == 6


class TestSettings:
    """Tests for Settings configuration class."""

    def test_defaults(self) -> None:
        settings = Settings()
        assert settings.default_timeout == 30
        assert settings.max_timeout == 300
        assert settings.default_memory == "256m"
        assert settings.max_memory_mb == 2048
        assert settings.default_cpu_shares == 512
        assert settings.max_cpu_cores == 4.0
        assert settings.default_disk == "512m"
        assert settings.max_disk_mb == 5000
        assert (
            settings.enable_network is False
            if hasattr(settings, "enable_network")
            else True
        )
        assert settings.auth_token is None
        assert settings.api_keys == []

    def test_auth_enabled_with_api_keys(self) -> None:
        settings = Settings(api_keys=["key1", "key2"])
        assert settings.auth_enabled is True

    def test_auth_enabled_with_token(self) -> None:
        settings = Settings(auth_token="my-token")
        assert settings.auth_enabled is True

    def test_auth_disabled_by_default(self) -> None:
        settings = Settings()
        assert settings.auth_enabled is False

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("LOG_LEVEL", "debug")
        monkeypatch.setenv("DEFAULT_TIMEOUT", "60")
        monkeypatch.setenv("MAX_TIMEOUT", "600")
        monkeypatch.setenv("CODE_RUNNER_API_KEYS", "key1,key2,key3")

        settings = Settings.from_env()
        assert settings.port == 9000
        assert settings.log_level == "debug"
        assert settings.default_timeout == 60
        assert settings.max_timeout == 600
        assert settings.api_keys == ["key1", "key2", "key3"]

    def test_from_env_empty_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CODE_RUNNER_API_KEYS", raising=False)
        monkeypatch.delenv("AUTH_TOKEN", raising=False)
        settings = Settings.from_env()
        assert settings.api_keys == []
        assert settings.auth_token is None

    def test_validate_valid_config(self) -> None:
        settings = Settings()
        # Should not raise
        settings.validate()

    def test_validate_invalid_timeout(self) -> None:
        settings = Settings(default_timeout=0)
        with pytest.raises(ValueError, match="DEFAULT_TIMEOUT must be positive"):
            settings.validate()

    def test_validate_max_less_than_default(self) -> None:
        settings = Settings(default_timeout=60, max_timeout=30)
        with pytest.raises(ValueError, match="MAX_TIMEOUT must be >= DEFAULT_TIMEOUT"):
            settings.validate()

    def test_validate_invalid_memory(self) -> None:
        settings = Settings(max_memory_mb=0)
        with pytest.raises(ValueError, match="MAX_MEMORY_MB must be positive"):
            settings.validate()

    def test_validate_invalid_cpu(self) -> None:
        settings = Settings(max_cpu_cores=0)
        with pytest.raises(ValueError, match="MAX_CPU_CORES must be positive"):
            settings.validate()

    def test_validate_invalid_disk(self) -> None:
        settings = Settings(max_disk_mb=-1)
        with pytest.raises(ValueError, match="MAX_DISK_MB must be positive"):
            settings.validate()

    def test_validate_invalid_cleanup_interval(self) -> None:
        settings = Settings(cleanup_interval_hours=0)
        with pytest.raises(ValueError, match="CLEANUP_INTERVAL_HOURS must be positive"):
            settings.validate()

    def test_validate_invalid_cleanup_age(self) -> None:
        settings = Settings(cleanup_max_age_days=-1)
        with pytest.raises(
            ValueError, match="CLEANUP_MAX_AGE_DAYS must be non-negative"
        ):
            settings.validate()
