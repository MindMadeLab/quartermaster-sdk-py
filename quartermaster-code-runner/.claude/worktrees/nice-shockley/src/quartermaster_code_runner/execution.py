"""Docker container execution logic.

Handles creating, running, and cleaning up Docker containers
for sandboxed code execution.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import tarfile
import time
import uuid
from typing import Any, Callable, Optional

import docker
import docker.models.containers
from docker.errors import ContainerError, ImageNotFound
from fastapi import HTTPException

from quartermaster_code_runner.config import (
    METADATA_FILE_PATH,
    PREBUILT_IMAGE_PREFIX,
    get_image_name,
)
from quartermaster_code_runner.schemas import PrebuildSpec

logger = logging.getLogger(__name__)


def get_docker_client(timeout: int = 600) -> docker.DockerClient:
    """Create a Docker client from environment."""
    return docker.from_env(timeout=timeout)


async def execute_code_in_container(
    docker_client: docker.DockerClient,
    image: str,
    environment: dict[str, str],
    timeout_seconds: int,
    mem_limit: str,
    cpu_shares: int,
    disk_limit: str,
    network_disabled: bool,
    prebuild_spec: Optional[PrebuildSpec] = None,
    ensure_prebuilt_fn: Optional[Callable[..., Any]] = None,
) -> tuple[str, str, dict[str, Any], Optional[dict[str, Any]]]:
    """Execute code in a Docker container.

    Args:
        docker_client: Docker client instance.
        image: Short or full image name.
        environment: Environment variables for the container.
        timeout_seconds: Hard timeout in seconds.
        mem_limit: Memory limit string (e.g., "256m").
        cpu_shares: CPU shares allocation.
        disk_limit: Disk limit for tmpfs mounts.
        network_disabled: Whether to disable networking.
        prebuild_spec: Optional prebuild specification.
        ensure_prebuilt_fn: Function to ensure prebuilt image exists.

    Returns:
        Tuple of (stdout, stderr, result_dict, metadata).
    """
    container = None
    stdout = ""
    stderr = ""
    result: dict[str, Any] = {}
    metadata = None
    full_image_name = get_image_name(image)

    # Handle prebuilt image rebuilds
    if image.startswith(PREBUILT_IMAGE_PREFIX) and prebuild_spec and ensure_prebuilt_fn:
        try:
            await asyncio.to_thread(
                ensure_prebuilt_fn,
                image,
                prebuild_spec,
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Prebuilt image '{full_image_name}' could not be rebuilt: {e}",
            )

    # Create a temporary volume for metadata exchange
    volume_name = f"quartermaster_metadata_{uuid.uuid4().hex[:12]}"
    volume = None

    try:
        volume = await asyncio.to_thread(
            docker_client.volumes.create,
            name=volume_name,
        )

        volumes_dict = {volume_name: {"bind": "/metadata", "mode": "rw"}}

        t_run = time.monotonic()
        container = await asyncio.to_thread(
            docker_client.containers.run,
            image=full_image_name,
            environment=environment,
            working_dir="/tmp",
            mem_limit=mem_limit,
            cpu_shares=cpu_shares,
            detach=True,
            network_disabled=network_disabled,
            read_only=True,
            tmpfs={
                "/tmp": f"size={disk_limit},exec",
                "/workspace": f"size={disk_limit},exec",
            },
            volumes=volumes_dict,
        )
        logger.info(
            "[exec] container started in %.2fs image=%s",
            time.monotonic() - t_run,
            full_image_name,
        )

        # Wait for container to finish
        t_wait = time.monotonic()
        wait_response = await asyncio.to_thread(container.wait, timeout=timeout_seconds)
        logger.info(
            "[exec] container finished in %.2fs exit=%s",
            time.monotonic() - t_wait,
            wait_response.get("StatusCode"),
        )
        result = {"StatusCode": wait_response["StatusCode"]}

        # Collect output
        stdout_bytes = await asyncio.to_thread(
            container.logs, stdout=True, stderr=False
        )
        stdout = stdout_bytes.decode("utf-8")
        stderr_bytes = await asyncio.to_thread(
            container.logs, stdout=False, stderr=True
        )
        stderr = stderr_bytes.decode("utf-8")

        # Read metadata from volume
        metadata = await _read_metadata(container)

    except ContainerError as e:
        if e.container:
            stderr_bytes = await asyncio.to_thread(
                e.container.logs, stdout=False, stderr=True
            )
            stderr = stderr_bytes.decode("utf-8")
            result = {"StatusCode": e.exit_status}
        else:
            stderr = str(e)
    except ImageNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"Image '{full_image_name}' not found. "
            f"Build it with: make build-runtime lang={image}",
        )
    except Exception as e:
        if container:
            with contextlib.suppress(Exception):
                stdout_bytes = await asyncio.to_thread(
                    container.logs, stdout=True, stderr=False
                )
                stdout = stdout_bytes.decode("utf-8")
            with contextlib.suppress(Exception):
                stderr_bytes = await asyncio.to_thread(
                    container.logs, stdout=False, stderr=True
                )
                stderr = stderr_bytes.decode("utf-8")
            with contextlib.suppress(Exception):
                await asyncio.to_thread(container.stop, timeout=2)
            result = {"StatusCode": -1}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred: {e}",
            )
    finally:
        if container:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(container.remove, v=True)
        if volume:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(volume.remove)

    return stdout, stderr, result, metadata


async def _read_metadata(
    container: docker.models.containers.Container,
) -> Optional[dict[str, Any]]:
    """Read metadata JSON from the container's metadata volume.

    Returns None if no metadata file exists.
    """
    try:
        bits, _stat = await asyncio.to_thread(container.get_archive, METADATA_FILE_PATH)
        tar_stream = io.BytesIO()
        for chunk in bits:
            tar_stream.write(chunk)
        tar_stream.seek(0)
        with tarfile.open(fileobj=tar_stream, mode="r") as tar:
            member = tar.getmembers()[0]
            f = tar.extractfile(member)
            if f:
                result: dict[str, Any] = json.loads(f.read().decode("utf-8"))
                return result
    except docker.errors.NotFound:
        pass
    except Exception as e:
        logger.warning(
            "[exec] metadata read failed: %s: %s",
            type(e).__name__,
            e,
        )
    return None


def cleanup_orphaned_containers(docker_client: docker.DockerClient) -> None:
    """Remove leftover metadata volumes from previous runs.

    Called on startup to prevent stale resources.
    """
    for v in docker_client.volumes.list(filters={"name": "quartermaster_metadata_"}):
        with contextlib.suppress(Exception):
            v.remove(force=True)
            logger.info("Removed orphaned metadata volume %s", v.name)
