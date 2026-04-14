"""Pydantic request/response models for code execution API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator

from quartermaster_code_runner.config import (
    DEFAULT_IMAGE,
    PREBUILT_IMAGE_PREFIX,
    SUPPORTED_IMAGES,
    get_short_image_name,
)


class PrebuildSpec(BaseModel):
    """Build recipe for a prebuilt image, used for on-the-fly rebuilds."""

    base_image: str
    setup_script: str


class CodeExecutionRequest(BaseModel):
    """Request model for code execution.

    Attributes:
        code: Source code to execute.
        image: Runtime image (python, node, go, rust, deno, bun).
        files: Additional files to include {filename: content}.
        entrypoint: Custom entrypoint command (overrides default).
        timeout: Execution timeout in seconds.
        mem_limit: Memory limit (e.g., "256m").
        cpu_shares: CPU shares (Docker cpu_shares).
        disk_limit: Disk limit for tmpfs (e.g., "512m").
        allow_network: Whether to allow outbound network access.
        environment: Environment variables to inject.
        prebuild_spec: Optional prebuild specification for on-the-fly rebuilds.
    """

    code: str
    image: str = DEFAULT_IMAGE
    files: Optional[dict[str, str]] = None
    entrypoint: Optional[str] = None
    timeout: Optional[int] = None
    mem_limit: Optional[str] = None
    cpu_shares: Optional[int] = None
    disk_limit: Optional[str] = None
    allow_network: bool = False
    environment: Optional[dict[str, str]] = None
    prebuild_spec: Optional[PrebuildSpec] = None

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str) -> str:
        short_name = get_short_image_name(v)
        if short_name not in SUPPORTED_IMAGES:
            full_prebuilt = (
                v
                if v.startswith(PREBUILT_IMAGE_PREFIX)
                else f"{PREBUILT_IMAGE_PREFIX}{v}"
            )
            return full_prebuilt
        return short_name

    @field_validator("environment")
    @classmethod
    def validate_environment(
        cls, v: Optional[dict[str, str]]
    ) -> Optional[dict[str, str]]:
        if v is None:
            return v
        reserved_keys = {
            "ENCODED_CODE",
            "ENCODED_FILES",
            "CUSTOM_ENTRYPOINT",
        }
        for key in v:
            if key in reserved_keys:
                raise ValueError(
                    f"Environment variable '{key}' is reserved and cannot be set."
                )
        return v


class CodeExecutionResponse(BaseModel):
    """Response model for code execution."""

    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    metadata: Optional[dict[str, object]] = None


class PrebuildRequest(BaseModel):
    """Request model for building a prebuilt image."""

    tag: str
    base_image: str = DEFAULT_IMAGE
    setup_script: str

    @field_validator("base_image")
    @classmethod
    def validate_base_image(cls, v: str) -> str:
        short_name = get_short_image_name(v)
        if short_name not in SUPPORTED_IMAGES:
            raise ValueError(
                f"Unsupported base image '{v}'. "
                f"Supported: {', '.join(SUPPORTED_IMAGES)}"
            )
        return short_name

    @field_validator("tag")
    @classmethod
    def validate_tag(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Tag cannot be empty.")
        if v.startswith(PREBUILT_IMAGE_PREFIX):
            v = v[len(PREBUILT_IMAGE_PREFIX) :]
        if not all(c.isalnum() or c in "-_." for c in v):
            raise ValueError(
                "Tag must contain only alphanumeric characters, "
                "hyphens, underscores, or dots."
            )
        return v


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    docker_connected: bool
    auth_enabled: bool


class RuntimeInfo(BaseModel):
    """Information about a single runtime."""

    id: str
    name: str
    description: str
    default_entrypoint: str
    file_extension: str
    main_file: str
    completions: list[dict[str, object]] = []


class RuntimesResponse(BaseModel):
    """Response for listing available runtimes."""

    images: list[RuntimeInfo]
    default: str
