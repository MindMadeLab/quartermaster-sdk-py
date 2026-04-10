"""qm-code-runner SDK for Python runtime."""

import json
import os
import urllib.error
import urllib.request

_METADATA_FILE = "/metadata/.qm_metadata.json"


def set_metadata(data):
    """
    Set the result metadata to be returned to the backend.

    This separates structured results from stdout/stderr logs.

    Args:
        data: Any JSON-serializable data (dict, list, str, int, etc.)

    Example:
        from sdk import set_metadata

        result = {"status": "success", "count": 42}
        set_metadata(result)
    """
    with open(_METADATA_FILE, "w") as f:
        json.dump(data, f)


def get_metadata():
    """
    Get previously set metadata (useful for reading/modifying).

    Returns:
        The previously set metadata, or None if not set.
    """
    if not os.path.exists(_METADATA_FILE):
        return None
    with open(_METADATA_FILE, "r") as f:
        return json.load(f)


def load_file(path):
    """Load a file from the flow's environment.

    Only available during flow execution, not test runs.

    Args:
        path: Path to the file within the environment.

    Returns:
        The file content as a string.

    Example:
        from sdk import load_file

        content = load_file("data/config.json")
    """
    webdav_url = os.environ.get("QM_WEBDAV_URL")
    if not webdav_url:
        raise RuntimeError(
            "load_file() is only available during flow execution. "
            "For test runs, use mounted environments instead."
        )
    url = webdav_url.rstrip("/") + "/" + path.lstrip("/")
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(f"File not found: {path}")
        raise RuntimeError(f"Failed to load file: {e}")


def save_file(path, content):
    """Save a file to the flow's environment.

    Only available during flow execution, not test runs.

    Args:
        path: Path to the file within the environment.
        content: The file content to save.

    Example:
        from sdk import save_file

        save_file("output/result.txt", "Hello, world!")
    """
    webdav_url = os.environ.get("QM_WEBDAV_URL")
    if not webdav_url:
        raise RuntimeError(
            "save_file() is only available during flow execution. "
            "For test runs, use mounted environments instead."
        )
    url = webdav_url.rstrip("/") + "/" + path.lstrip("/")
    data = content.encode("utf-8")
    req = urllib.request.Request(url, data=data, method="PUT")
    req.add_header("Content-Type", "application/octet-stream")
    try:
        urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to save file: {e}")
