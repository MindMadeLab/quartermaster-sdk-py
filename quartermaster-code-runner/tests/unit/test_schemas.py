"""Unit tests for Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from quartermaster_code_runner.schemas import (
    CodeExecutionRequest,
    CodeExecutionResponse,
    HealthResponse,
    PrebuildRequest,
)


class TestCodeExecutionRequest:
    """Tests for CodeExecutionRequest validation."""

    def test_minimal_request(self) -> None:
        req = CodeExecutionRequest(code="print('hello')")
        assert req.code == "print('hello')"
        assert req.image == "python"

    def test_default_image_is_python(self) -> None:
        req = CodeExecutionRequest(code="print(1)")
        assert req.image == "python"

    def test_valid_images(self) -> None:
        for lang in ["python", "node", "go", "rust", "deno", "bun"]:
            req = CodeExecutionRequest(code="x", image=lang)
            assert req.image == lang

    def test_prebuilt_image_prefix(self) -> None:
        req = CodeExecutionRequest(code="x", image="prebuilt-myimg")
        assert req.image == "prebuilt-myimg"

    def test_unsupported_image_becomes_prebuilt(self) -> None:
        req = CodeExecutionRequest(code="x", image="unknown-lang")
        assert req.image == "prebuilt-unknown-lang"

    def test_with_files(self) -> None:
        req = CodeExecutionRequest(
            code="import a",
            files={"a.py": "x = 42"},
        )
        assert req.files == {"a.py": "x = 42"}

    def test_with_entrypoint(self) -> None:
        req = CodeExecutionRequest(
            code="",
            entrypoint="python -c 'print(1)'",
        )
        assert req.entrypoint == "python -c 'print(1)'"

    def test_with_environment(self) -> None:
        req = CodeExecutionRequest(
            code="x",
            environment={"MY_VAR": "value"},
        )
        assert req.environment == {"MY_VAR": "value"}

    def test_reserved_env_var_rejected(self) -> None:
        with pytest.raises(ValidationError, match="reserved"):
            CodeExecutionRequest(
                code="x",
                environment={"ENCODED_CODE": "bad"},
            )

    def test_reserved_env_var_encoded_files(self) -> None:
        with pytest.raises(ValidationError, match="reserved"):
            CodeExecutionRequest(
                code="x",
                environment={"ENCODED_FILES": "bad"},
            )

    def test_reserved_env_var_custom_entrypoint(self) -> None:
        with pytest.raises(ValidationError, match="reserved"):
            CodeExecutionRequest(
                code="x",
                environment={"CUSTOM_ENTRYPOINT": "bad"},
            )

    def test_with_resource_limits(self) -> None:
        req = CodeExecutionRequest(
            code="x",
            timeout=10,
            mem_limit="128m",
            cpu_shares=256,
            disk_limit="100m",
        )
        assert req.timeout == 10
        assert req.mem_limit == "128m"
        assert req.cpu_shares == 256
        assert req.disk_limit == "100m"

    def test_network_disabled_by_default(self) -> None:
        # Security default: deny outbound network unless explicitly enabled.
        # (Was True historically; flipped during the security-hardening pass.)
        req = CodeExecutionRequest(code="x")
        assert req.allow_network is False

    def test_network_can_be_enabled(self) -> None:
        req = CodeExecutionRequest(code="x", allow_network=True)
        assert req.allow_network is True


class TestCodeExecutionResponse:
    """Tests for CodeExecutionResponse."""

    def test_basic_response(self) -> None:
        resp = CodeExecutionResponse(
            stdout="hello\n",
            stderr="",
            exit_code=0,
            execution_time=0.5,
        )
        assert resp.stdout == "hello\n"
        assert resp.exit_code == 0
        assert resp.metadata is None

    def test_response_with_metadata(self) -> None:
        resp = CodeExecutionResponse(
            stdout="",
            stderr="",
            exit_code=0,
            execution_time=1.0,
            metadata={"key": "value"},
        )
        assert resp.metadata == {"key": "value"}


class TestPrebuildRequest:
    """Tests for PrebuildRequest validation."""

    def test_valid_prebuild(self) -> None:
        req = PrebuildRequest(
            tag="my-image",
            base_image="python",
            setup_script="pip install numpy",
        )
        assert req.tag == "my-image"
        assert req.base_image == "python"

    def test_empty_tag_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            PrebuildRequest(
                tag="",
                base_image="python",
                setup_script="echo hi",
            )

    def test_invalid_tag_chars_rejected(self) -> None:
        with pytest.raises(ValidationError, match="alphanumeric"):
            PrebuildRequest(
                tag="my image!",
                base_image="python",
                setup_script="echo hi",
            )

    def test_tag_strips_prebuilt_prefix(self) -> None:
        req = PrebuildRequest(
            tag="prebuilt-myimg",
            base_image="python",
            setup_script="echo hi",
        )
        assert req.tag == "myimg"

    def test_unsupported_base_image_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unsupported"):
            PrebuildRequest(
                tag="test",
                base_image="ruby",
                setup_script="echo hi",
            )

    def test_valid_base_images(self) -> None:
        for lang in ["python", "node", "go", "rust", "deno", "bun"]:
            req = PrebuildRequest(
                tag="test",
                base_image=lang,
                setup_script="echo hi",
            )
            assert req.base_image == lang


class TestHealthResponse:
    """Tests for HealthResponse."""

    def test_health_ok(self) -> None:
        resp = HealthResponse(
            status="ok",
            docker_connected=True,
            auth_enabled=True,
        )
        assert resp.status == "ok"
        assert resp.docker_connected is True
