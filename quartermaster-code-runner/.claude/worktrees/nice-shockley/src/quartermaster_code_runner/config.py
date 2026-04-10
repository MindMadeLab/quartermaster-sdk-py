"""Configuration for code runner service.

All settings are configurable via environment variables with sensible defaults.
Supports .env file loading via python-dotenv.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Image naming conventions
RUNTIME_IMAGE_PREFIX = "code-runner-"
PREBUILT_IMAGE_PREFIX = "prebuilt-"
DEFAULT_IMAGE = "python"
SUPPORTED_IMAGES = ["python", "node", "bun", "go", "deno", "rust"]
METADATA_FILE_PATH = "/metadata/.quartermaster_metadata.json"
MAX_CODE_SIZE_BYTES = 1024 * 1024  # 1MB


def get_image_name(image: str) -> str:
    """Convert short image name to full Docker image name."""
    if image.startswith(RUNTIME_IMAGE_PREFIX) or image.startswith(
        PREBUILT_IMAGE_PREFIX
    ):
        return image
    return f"{RUNTIME_IMAGE_PREFIX}{image}"


def get_short_image_name(image: str) -> str:
    """Get short image name from full or short name."""
    if image.startswith(RUNTIME_IMAGE_PREFIX):
        return image[len(RUNTIME_IMAGE_PREFIX) :]
    return image


@dataclass
class ResourceLimits:
    """Resource limits for a single code execution."""

    cpu_cores: float = 1.0
    memory_mb: int = 512
    disk_mb: int = 500


@dataclass
class Settings:
    """Global service settings loaded from environment variables.

    All settings have sensible defaults and can be overridden via
    environment variables or a .env file.
    """

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Execution defaults
    default_timeout: int = 30
    max_timeout: int = 300
    default_memory: str = "256m"
    max_memory_mb: int = 2048
    default_cpu_shares: int = 512
    max_cpu_cores: float = 4.0
    default_disk: str = "512m"
    max_disk_mb: int = 5000

    # Docker
    docker_socket: str = "/var/run/docker.sock"

    # Security
    auth_token: str | None = None
    api_keys: list[str] = field(default_factory=list)

    # Cleanup
    cleanup_interval_hours: int = 24
    cleanup_max_age_days: int = 7

    # Runtime image build path (relative to package)
    runtime_dir: str = ""

    @classmethod
    def from_env(cls, env_file: str | None = None) -> Settings:
        """Load settings from environment variables, with optional .env file.

        Args:
            env_file: Path to .env file. If None, searches current directory.
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        api_keys_str = os.getenv("CODE_RUNNER_API_KEYS", "")
        api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]

        # Also support AUTH_TOKEN as single-token auth
        auth_token = os.getenv("AUTH_TOKEN")

        # Determine runtime directory
        runtime_dir = os.getenv(
            "RUNTIME_DIR",
            str(Path(__file__).parent / "runtime"),
        )

        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "info"),
            default_timeout=int(os.getenv("DEFAULT_TIMEOUT", "30")),
            max_timeout=int(os.getenv("MAX_TIMEOUT", "300")),
            default_memory=os.getenv("DEFAULT_MEMORY", "256m"),
            max_memory_mb=int(os.getenv("MAX_MEMORY_MB", "2048")),
            default_cpu_shares=int(os.getenv("DEFAULT_CPU_SHARES", "512")),
            max_cpu_cores=float(os.getenv("MAX_CPU_CORES", "4.0")),
            default_disk=os.getenv("DEFAULT_DISK", "512m"),
            max_disk_mb=int(os.getenv("MAX_DISK_MB", "5000")),
            docker_socket=os.getenv("DOCKER_SOCKET", "/var/run/docker.sock"),
            auth_token=auth_token,
            api_keys=api_keys,
            cleanup_interval_hours=int(os.getenv("CLEANUP_INTERVAL_HOURS", "24")),
            cleanup_max_age_days=int(os.getenv("CLEANUP_MAX_AGE_DAYS", "7")),
            runtime_dir=runtime_dir,
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings/errors.

        Raises:
            ValueError: If configuration is fatally invalid.
        """
        errors: list[str] = []

        if self.default_timeout <= 0:
            errors.append("DEFAULT_TIMEOUT must be positive")
        if self.max_timeout < self.default_timeout:
            errors.append("MAX_TIMEOUT must be >= DEFAULT_TIMEOUT")
        if self.max_memory_mb <= 0:
            errors.append("MAX_MEMORY_MB must be positive")
        if self.max_cpu_cores <= 0:
            errors.append("MAX_CPU_CORES must be positive")
        if self.max_disk_mb <= 0:
            errors.append("MAX_DISK_MB must be positive")
        if self.cleanup_interval_hours <= 0:
            errors.append("CLEANUP_INTERVAL_HOURS must be positive")
        if self.cleanup_max_age_days < 0:
            errors.append("CLEANUP_MAX_AGE_DAYS must be non-negative")

        if not Path(self.docker_socket).exists():
            logger.warning("Docker socket not found at %s", self.docker_socket)

        if not self.api_keys and not self.auth_token:
            logger.warning(
                "No authentication configured. API is open to all requests. "
                "Set CODE_RUNNER_API_KEYS or AUTH_TOKEN to secure the API."
            )

        if errors:
            raise ValueError(
                "Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return []

    @property
    def auth_enabled(self) -> bool:
        """Whether any authentication method is configured."""
        return bool(self.api_keys) or bool(self.auth_token)
