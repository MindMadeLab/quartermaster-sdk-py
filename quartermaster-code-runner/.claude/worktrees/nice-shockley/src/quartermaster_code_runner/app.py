"""FastAPI application for sandboxed code execution.

Provides REST API endpoints for executing code in isolated Docker containers
with support for Python, Node.js, Go, Rust, Deno, and Bun.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tarfile
import time
from typing import Any
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from quartermaster_code_runner.config import (
    MAX_CODE_SIZE_BYTES,
    Settings,
)
from quartermaster_code_runner.execution import (
    cleanup_orphaned_containers,
    execute_code_in_container,
    get_docker_client,
)
from quartermaster_code_runner.images import (
    cleanup_prebuilds,
    configure_images,
    ensure_prebuilt_image,
    router as images_router,
)
from quartermaster_code_runner.schemas import CodeExecutionRequest, CodeExecutionResponse
from quartermaster_code_runner.security import configure_auth, verify_auth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Global settings and docker client
_settings: Settings | None = None
_docker_client = None


async def _periodic_cleanup(interval_hours: int, max_age_days: int) -> None:
    """Periodically clean up old prebuilt images."""
    while True:
        await asyncio.sleep(interval_hours * 3600)
        try:
            removed = await asyncio.to_thread(cleanup_prebuilds, max_age_days)
            if removed:
                logger.info(
                    "Auto-cleanup removed %d prebuilt images: %s",
                    len(removed),
                    removed,
                )
        except Exception:
            logger.exception("Auto-cleanup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Application lifespan: setup and teardown."""
    global _settings, _docker_client

    # Load settings
    _settings = Settings.from_env()
    _settings.validate()

    # Set log level
    log_level = getattr(logging, _settings.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    # Configure authentication
    configure_auth(
        api_keys=_settings.api_keys,
        auth_token=_settings.auth_token,
    )

    # Initialize Docker client
    _docker_client = get_docker_client()

    # Configure image management
    configure_images(
        docker_client=_docker_client,
        runtime_dir=_settings.runtime_dir,
        verify_auth_dep=verify_auth,
    )

    # Cleanup orphaned resources from previous runs
    try:
        cleanup_orphaned_containers(_docker_client)
        logger.info("Startup: orphaned resources cleaned up")
    except Exception:
        logger.exception("Startup: failed to clean orphaned resources")

    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(
        _periodic_cleanup(
            _settings.cleanup_interval_hours,
            _settings.cleanup_max_age_days,
        )
    )

    logger.info(
        "Code Runner started (auth=%s, runtime_dir=%s)",
        "enabled" if _settings.auth_enabled else "disabled",
        _settings.runtime_dir,
    )

    try:
        yield
    finally:
        cleanup_task.cancel()
        logger.info("Code Runner shutting down")


app = FastAPI(
    title="quartermaster-code-runner",
    description="Secure sandboxed code execution service",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(images_router)


@app.post("/run", response_model=CodeExecutionResponse)
async def run_code(
    request: Request,
    payload: CodeExecutionRequest,
    _: str | None = Depends(verify_auth),
) -> dict[str, Any]:
    """Execute code in a secure, isolated Docker container.

    Supports Python, Node.js, Go, Rust, Deno, and Bun runtimes.
    """
    assert _settings is not None
    assert _docker_client is not None

    logger.info("[run] request image=%s", payload.image)

    if not payload.code and not payload.entrypoint:
        raise HTTPException(
            status_code=400,
            detail="Either code or entrypoint must be provided.",
        )

    # Check code size
    total_size = len(payload.code.encode("utf-8"))
    if payload.files:
        for content in payload.files.values():
            total_size += len(content.encode("utf-8"))

    if total_size > MAX_CODE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Code size exceeds the limit.")

    start_time = time.time()

    # Apply defaults from settings
    timeout_seconds = (
        payload.timeout if payload.timeout is not None else _settings.default_timeout
    )
    mem_limit = (
        payload.mem_limit if payload.mem_limit is not None else _settings.default_memory
    )
    cpu_shares = (
        payload.cpu_shares
        if payload.cpu_shares is not None
        else _settings.default_cpu_shares
    )
    disk_limit = (
        payload.disk_limit if payload.disk_limit is not None else _settings.default_disk
    )
    network_disabled = not payload.allow_network

    # Build container environment
    container_env: dict[str, str] = {"HOME": "/tmp"}
    if payload.environment:
        container_env.update(payload.environment)
    if payload.code:
        container_env["ENCODED_CODE"] = base64.b64encode(
            payload.code.encode("utf-8")
        ).decode("utf-8")
    if payload.entrypoint:
        container_env["CUSTOM_ENTRYPOINT"] = payload.entrypoint

    # Package additional files as tar archive
    if payload.files:
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            for filename, content in payload.files.items():
                if ".." in filename or os.path.isabs(filename):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid filename: {filename}",
                    )
                tarinfo = tarfile.TarInfo(name=filename)
                file_bytes = content.encode("utf-8")
                tarinfo.size = len(file_bytes)
                tar.addfile(tarinfo, io.BytesIO(file_bytes))

        tar_buffer.seek(0)
        encoded_files = base64.b64encode(tar_buffer.read()).decode("utf-8")
        container_env["ENCODED_FILES"] = encoded_files

    # Execute
    stdout, stderr, result, metadata = await execute_code_in_container(
        docker_client=_docker_client,
        image=payload.image,
        environment=container_env,
        timeout_seconds=timeout_seconds,
        mem_limit=mem_limit,
        cpu_shares=cpu_shares,
        disk_limit=disk_limit,
        network_disabled=network_disabled,
        prebuild_spec=payload.prebuild_spec,
        ensure_prebuilt_fn=ensure_prebuilt_image,
    )

    execution_time = time.time() - start_time

    response: dict[str, Any] = {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": result.get("StatusCode", -1),
        "execution_time": round(execution_time, 4),
    }

    if metadata is not None:
        response["metadata"] = metadata

    return response


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint."""
    docker_connected = False
    if _docker_client:
        try:
            _docker_client.ping()
            docker_connected = True
        except Exception:
            pass

    return {
        "status": "ok" if docker_connected else "degraded",
        "docker_connected": docker_connected,
        "auth_enabled": _settings.auth_enabled if _settings else False,
    }


@app.get("/runtimes")
def list_runtimes() -> dict[str, Any]:
    """List available runtimes. Alias for /images."""
    from quartermaster_code_runner.images import list_images

    return list_images()


@app.get("/")
def read_root() -> dict[str, Any]:
    """Root endpoint."""
    return {"message": "Code Runner is operational."}
