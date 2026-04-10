"""Docker image management for runtime and prebuilt images.

Handles building, listing, and cleaning up runtime and prebuilt Docker images.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Optional

import docker
from docker.errors import BuildError, ImageNotFound
from fastapi import APIRouter, HTTPException, Query

from qm_code_runner.config import (
    DEFAULT_IMAGE,
    PREBUILT_IMAGE_PREFIX,
    SUPPORTED_IMAGES,
    get_image_name,
    get_short_image_name,
)
from qm_code_runner.schemas import PrebuildRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level docker client, set during app startup
_docker_client: Optional[docker.DockerClient] = None
_runtime_dir: str = ""
_verify_auth: Optional[object] = None


def configure_images(
    docker_client: docker.DockerClient,
    runtime_dir: str,
    verify_auth_dep: object,
) -> None:
    """Configure image management module.

    Args:
        docker_client: Docker client instance.
        runtime_dir: Path to runtime directory containing Dockerfiles.
        verify_auth_dep: FastAPI dependency for auth verification.
    """
    global _docker_client, _runtime_dir, _verify_auth
    _docker_client = docker_client
    _runtime_dir = runtime_dir
    _verify_auth = verify_auth_dep


def _get_client() -> docker.DockerClient:
    if _docker_client is None:
        raise RuntimeError(
            "Image management not configured. Call configure_images() first."
        )
    return _docker_client


def _safe_list_images() -> list[Any]:
    """List Docker images, handling race conditions."""
    client = _get_client()
    images = []
    for img_summary in client.api.images():
        img_id = img_summary.get("Id", "")
        try:
            images.append(client.images.get(img_id))
        except ImageNotFound:
            continue
    return images


def cleanup_prebuilds(max_age_days: int = 7) -> list[str]:
    """Remove prebuilt images older than max_age_days."""
    client = _get_client()
    now = datetime.now(timezone.utc)
    removed: list[str] = []
    targets: list[str] = []

    for img in _safe_list_images():
        for tag in img.tags or []:
            tag_name = tag.split(":")[0]
            if not tag_name.startswith(PREBUILT_IMAGE_PREFIX):
                continue
            created_str = img.attrs.get("Created", "")
            if not created_str:
                continue
            created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            age_days = (now - created).total_seconds() / 86400
            if age_days >= max_age_days:
                targets.append(tag_name)

    for tag_name in targets:
        with contextlib.suppress(Exception):
            client.images.remove(image=tag_name, force=True)
            removed.append(tag_name)

    return removed


def ensure_runtime_image(short_name: str) -> bool:
    """Build a single runtime image if it's missing. Returns True if rebuilt."""
    client = _get_client()
    full_name = get_image_name(short_name)
    try:
        client.images.get(full_name)
        return False
    except ImageNotFound:
        pass

    build_path = os.path.join(_runtime_dir, short_name)
    if not os.path.exists(build_path):
        return False

    logger.info("Auto-building missing runtime image '%s'...", full_name)
    client.images.build(
        path=build_path,
        dockerfile="Dockerfile",
        tag=full_name,
        rm=True,
    )
    logger.info("Runtime image '%s' built successfully.", full_name)
    return True


def ensure_prebuilt_image(image_name: str, spec: PrebuildRequest | object) -> bool:
    """Rebuild a prebuilt image on the fly if it's missing."""
    client = _get_client()
    try:
        client.images.get(image_name)
        return False
    except ImageNotFound:
        pass

    base_image = getattr(spec, "base_image", "python")
    setup_script = getattr(spec, "setup_script", "")

    tag = image_name.removeprefix(PREBUILT_IMAGE_PREFIX)
    logger.info("Auto-rebuilding missing prebuilt image '%s'...", image_name)
    ensure_runtime_image(get_short_image_name(base_image))
    _build_prebuilt_image(tag, base_image, setup_script)
    logger.info("Prebuilt image '%s' rebuilt successfully.", image_name)
    return True


def build_runtime_images() -> None:
    """Build all runtime Docker images."""
    client = _get_client()
    for image_name in SUPPORTED_IMAGES:
        full_name = get_image_name(image_name)
        build_path = os.path.join(_runtime_dir, image_name)

        if not os.path.exists(build_path):
            logger.warning(
                "Skipping '%s' - directory not found at %s",
                image_name,
                build_path,
            )
            continue

        try:
            client.images.remove(image=full_name, force=True)
            logger.info("Removed old runtime image '%s'.", full_name)
        except ImageNotFound:
            pass

        logger.info("Building runtime image '%s'...", full_name)
        try:
            client.images.build(
                path=build_path,
                dockerfile="Dockerfile",
                tag=full_name,
                rm=True,
            )
            logger.info("Runtime image '%s' built successfully.", full_name)
        except BuildError as e:
            logger.error("Failed to build runtime image '%s': %s", full_name, e)
            raise


# =============================================================================
# Prebuilt image helpers
# =============================================================================


def _get_prebuilt_image_tag(tag: str) -> str:
    if tag.startswith(PREBUILT_IMAGE_PREFIX):
        return tag
    return f"{PREBUILT_IMAGE_PREFIX}{tag}"


def _get_image_size_bytes(image_name: str) -> int:
    client = _get_client()
    try:
        img = client.images.get(image_name)
        return img.attrs.get("Size", 0)  # type: ignore[no-any-return]
    except ImageNotFound:
        return 0


def _build_prebuilt_image(
    tag: str, base_image: str, setup_script: str
) -> dict[str, Any]:
    client = _get_client()
    full_tag = _get_prebuilt_image_tag(tag)
    base_full = get_image_name(base_image)

    dockerfile_content = (
        f"FROM {base_full}\n"
        f"COPY setup.sh /tmp/setup.sh\n"
        f"RUN sh /tmp/setup.sh && rm /tmp/setup.sh\n"
    )

    with contextlib.suppress(ImageNotFound):
        client.images.remove(image=full_tag, force=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile_path = os.path.join(tmpdir, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        setup_path = os.path.join(tmpdir, "setup.sh")
        with open(setup_path, "w") as f:
            f.write(setup_script)

        client.images.build(
            path=tmpdir,
            dockerfile="Dockerfile",
            tag=full_tag,
            rm=True,
        )

    size_bytes = _get_image_size_bytes(full_tag)
    base_size = _get_image_size_bytes(base_full)
    layer_size = max(size_bytes - base_size, 0)

    return {
        "tag": full_tag,
        "base_image": base_full,
        "size_bytes": size_bytes,
        "layer_size_bytes": layer_size,
        "status": "ready",
    }


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/images")
def list_images() -> dict[str, Any]:
    """List available runtime images with metadata from Docker labels."""
    client = _get_client()
    images = []

    for short_name in SUPPORTED_IMAGES:
        full_name = get_image_name(short_name)
        image_data: dict[str, Any] = {
            "id": full_name,
            "name": short_name.capitalize(),
            "description": f"{short_name.capitalize()} runtime",
            "default_entrypoint": f"{short_name} main.py",
            "file_extension": ".py",
            "main_file": "main.py",
            "completions": [],
        }

        try:
            image = client.images.get(full_name)
            labels = image.labels or {}
            for label_key, data_key in [
                ("qm.name", "name"),
                ("qm.description", "description"),
                ("qm.default_entrypoint", "default_entrypoint"),
                ("qm.file_extension", "file_extension"),
                ("qm.main_file", "main_file"),
            ]:
                if label_key in labels:
                    image_data[data_key] = labels[label_key]
        except ImageNotFound:
            pass

        completions_path = os.path.join(_runtime_dir, short_name, "completions.json")
        if os.path.exists(completions_path):
            with open(completions_path) as f:
                image_data["completions"] = json.load(f)

        images.append(image_data)

    return {
        "images": images,
        "default": get_image_name(DEFAULT_IMAGE),
    }


@router.post("/prebuild")
async def prebuild_image(payload: PrebuildRequest) -> dict[str, Any]:
    """Build a prebuilt Docker image extending a base runtime."""
    import asyncio

    try:
        result = await asyncio.to_thread(
            _build_prebuilt_image,
            payload.tag,
            payload.base_image,
            payload.setup_script,
        )
        return result
    except BuildError as e:
        build_log = ""
        if hasattr(e, "build_log"):
            for chunk in e.build_log:
                if "stream" in chunk:
                    build_log += chunk["stream"]
                elif "error" in chunk:
                    build_log += chunk["error"]
        raise HTTPException(
            status_code=400,
            detail=f"Build failed: {build_log or str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prebuild failed: {e}",
        )


@router.get("/prebuilds")
def list_prebuilds() -> dict[str, Any]:
    """List all prebuilt images with their sizes."""
    prebuilds: list[dict[str, Any]] = []
    for img in _safe_list_images():
        for tag in img.tags or []:
            tag_name = tag.split(":")[0]
            if tag_name.startswith(PREBUILT_IMAGE_PREFIX):
                labels = img.labels or {}
                prebuilds.append(
                    {
                        "tag": tag_name,
                        "size_bytes": img.attrs.get("Size", 0),
                        "created": img.attrs.get("Created", ""),
                        "labels": labels,
                    }
                )
    return {"prebuilds": prebuilds}


@router.get("/prebuilds/{tag}")
def get_prebuild(tag: str) -> dict[str, Any]:
    """Get details of a specific prebuilt image."""
    client = _get_client()
    full_tag = _get_prebuilt_image_tag(tag)
    try:
        img = client.images.get(full_tag)
        return {
            "tag": full_tag,
            "size_bytes": img.attrs.get("Size", 0),
            "created": img.attrs.get("Created", ""),
            "labels": img.labels or {},
        }
    except ImageNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"Prebuilt image '{full_tag}' not found.",
        )


@router.delete("/prebuilds/{tag}")
def delete_prebuild(tag: str) -> dict[str, Any]:
    """Delete a prebuilt image."""
    client = _get_client()
    full_tag = _get_prebuilt_image_tag(tag)
    try:
        client.images.remove(image=full_tag, force=True)
        return {"status": "deleted", "tag": full_tag}
    except ImageNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"Prebuilt image '{full_tag}' not found.",
        )


@router.post("/prebuilds/cleanup")
async def cleanup_prebuilt_images(
    max_age_days: int = Query(default=7, ge=0),
) -> dict[str, Any]:
    """Remove prebuilt images not used in max_age_days."""
    import asyncio

    removed = await asyncio.to_thread(cleanup_prebuilds, max_age_days)
    return {"removed": removed, "count": len(removed)}
