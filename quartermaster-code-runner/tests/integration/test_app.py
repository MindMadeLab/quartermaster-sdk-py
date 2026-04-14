"""Integration tests for the code execution API.

These tests require Docker to be running and runtime images to be built.
Build them with: make build-runtimes

The whole module is marked ``integration`` so it's automatically excluded
by CI's ``pytest -m "not integration"`` (Docker is too slow / heavy for
the per-PR test gate). Run them locally with::

    pytest -m integration         # only integration
    pytest                        # everything
    pytest -m "not integration"   # CI default
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from filelock import FileLock

from quartermaster_code_runner.images import build_runtime_images, configure_images
from quartermaster_code_runner.execution import get_docker_client

# Module-level marker — applies the ``integration`` mark to every test in
# this file. The marker is registered in pyproject.toml.
pytestmark = pytest.mark.integration


TEST_API_KEYS = "test-secret-key-1,test-secret-key-2"
FIRST_TEST_KEY = TEST_API_KEYS.split(",")[0]

# Check if Docker is available
_docker_available = False
try:
    client = get_docker_client()
    client.ping()
    _docker_available = True
except Exception:
    pass

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _docker_available,
        reason="Docker is not available",
    ),
]


@pytest.fixture(scope="session", autouse=True)
def build_docker_images(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Build all runtime Docker images once across all test workers."""
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    lock = root_tmp_dir / "build.lock"
    marker = root_tmp_dir / "build.done"

    docker_client = get_docker_client()
    runtime_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "quartermaster_code_runner",
        "runtime",
    )
    runtime_dir = os.path.abspath(runtime_dir)

    configure_images(
        docker_client=docker_client,
        runtime_dir=runtime_dir,
        verify_auth_dep=None,
    )

    with FileLock(str(lock)):
        if not marker.exists():
            build_runtime_images()
            marker.touch()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create authenticated test client."""
    monkeypatch.setenv("CODE_RUNNER_API_KEYS", TEST_API_KEYS)
    runtime_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "quartermaster_code_runner",
        "runtime",
    )
    monkeypatch.setenv("RUNTIME_DIR", os.path.abspath(runtime_dir))

    from quartermaster_code_runner.app import app

    with TestClient(app) as c:
        c.headers = {"X-API-Key": FIRST_TEST_KEY}  # type: ignore[assignment]
        yield c


# =============================================================================
# Basic API Tests
# =============================================================================


class TestBasicAPI:
    """Tests for basic API endpoints."""

    def test_root(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Code Runner is operational."}

    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["docker_connected"] is True

    def test_list_images(self, client: TestClient) -> None:
        response = client.get("/images")
        assert response.status_code == 200
        data = response.json()
        assert "images" in data
        assert "default" in data
        image_ids = [img["id"] for img in data["images"]]
        assert "code-runner-python" in image_ids
        assert "code-runner-node" in image_ids

    def test_list_runtimes(self, client: TestClient) -> None:
        response = client.get("/runtimes")
        assert response.status_code == 200
        data = response.json()
        assert "images" in data


# =============================================================================
# Python Runtime Tests
# =============================================================================


class TestPythonRuntime:
    """Tests for Python runtime execution."""

    def test_simple_code(self, client: TestClient) -> None:
        response = client.post("/run", json={"code": "print('hello world')"})
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "hello world\n"
        assert data["stderr"] == ""
        assert data["exit_code"] == 0

    def test_with_imports(self, client: TestClient) -> None:
        payload = {
            "files": {"a.py": "x = 42"},
            "code": "import a\nprint(a.x)",
        }
        response = client.post("/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "42\n"
        assert data["exit_code"] == 0

    def test_with_nested_imports(self, client: TestClient) -> None:
        payload = {
            "files": {
                "utils/helpers.py": "def get_message(): return 'nested hello'",
                "main_module.py": "from utils.helpers import get_message\n\ndef run(): return get_message()",
            },
            "code": "import main_module\nprint(main_module.run())",
        }
        response = client.post("/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "nested hello\n"
        assert data["exit_code"] == 0

    def test_stderr_and_exit_code(self, client: TestClient) -> None:
        code = "import sys; sys.stderr.write('error message'); sys.exit(1)"
        response = client.post("/run", json={"code": code})
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == ""
        assert data["stderr"] == "error message"
        assert data["exit_code"] == 1

    def test_environment_variables(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={
                "code": "import os; print(os.environ.get('SECRET_KEY'))",
                "environment": {"SECRET_KEY": "my_secret_123"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "my_secret_123\n"
        assert data["exit_code"] == 0

    def test_multiple_env_vars(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={
                "code": "import os; print(os.environ.get('VAR1'), os.environ.get('VAR2'))",
                "environment": {"VAR1": "value1", "VAR2": "value2"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "value1 value2\n"

    def test_sdk_metadata(self, client: TestClient) -> None:
        code = """
from sdk import set_metadata
print("Starting")
result = {"status": "success", "count": 42, "items": ["a", "b"]}
set_metadata(result)
print("Done")
"""
        response = client.post("/run", json={"code": code, "image": "python"})
        assert response.status_code == 200
        data = response.json()
        assert data["exit_code"] == 0
        assert "Starting" in data["stdout"]
        assert "metadata" in data
        assert data["metadata"]["status"] == "success"
        assert data["metadata"]["count"] == 42


# =============================================================================
# Node.js Runtime Tests
# =============================================================================


class TestNodeRuntime:
    """Tests for Node.js runtime."""

    def test_simple_code(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={
                "code": "console.log('hello from node')",
                "image": "node",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "hello from node\n"
        assert data["exit_code"] == 0

    def test_with_files(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={
                "code": "const helper = require('./helper.js'); console.log(helper.getMessage());",
                "image": "node",
                "files": {
                    "helper.js": "module.exports = { getMessage: () => 'from helper' };"
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "from helper\n"
        assert data["exit_code"] == 0

    def test_with_environment(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={
                "code": "console.log(process.env.MY_VAR)",
                "image": "node",
                "environment": {"MY_VAR": "test_value"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "test_value\n"

    def test_sdk_metadata(self, client: TestClient) -> None:
        code = """
const { setMetadata } = require('./sdk');
console.log("Starting");
setMetadata({ message: "hello", numbers: [1, 2, 3] });
console.log("Done");
"""
        response = client.post("/run", json={"code": code, "image": "node"})
        assert response.status_code == 200
        data = response.json()
        assert data["exit_code"] == 0
        assert "metadata" in data
        assert data["metadata"]["message"] == "hello"


# =============================================================================
# Go Runtime Tests
# =============================================================================


class TestGoRuntime:
    """Tests for Go runtime."""

    def test_hello_world(self, client: TestClient) -> None:
        code = 'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("hello from go")\n}\n'
        response = client.post("/run", json={"code": code, "image": "go"})
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "hello from go\n"
        assert data["exit_code"] == 0

    def test_with_environment(self, client: TestClient) -> None:
        code = 'package main\n\nimport (\n\t"fmt"\n\t"os"\n)\n\nfunc main() {\n\tsecret := os.Getenv("SECRET_API_KEY")\n\tfmt.Println(secret)\n}\n'
        response = client.post(
            "/run",
            json={
                "code": code,
                "image": "go",
                "environment": {"SECRET_API_KEY": "go_secret_123"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "go_secret_123\n"


# =============================================================================
# Deno Runtime Tests
# =============================================================================


class TestDenoRuntime:
    """Tests for Deno runtime."""

    def test_hello_world(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={
                "code": "const msg: string = 'hello from deno'; console.log(msg);",
                "image": "deno",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "hello from deno\n"
        assert data["exit_code"] == 0

    def test_with_environment(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={
                "code": "console.log(Deno.env.get('SECRET_TOKEN'))",
                "image": "deno",
                "environment": {"SECRET_TOKEN": "deno_secret_456"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "deno_secret_456\n"


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurity:
    """Tests for security features."""

    def test_filesystem_readonly(self, client: TestClient) -> None:
        code = "with open('/app/test.txt', 'w') as f: f.write('data')"
        response = client.post("/run", json={"code": code})
        assert response.status_code == 200
        data = response.json()
        assert data["exit_code"] != 0
        assert (
            "Read-only file system" in data["stderr"]
            or "Permission denied" in data["stderr"]
        )

    def test_network_allowed_by_default(self, client: TestClient) -> None:
        code = (
            "import json, os\nprint(json.dumps(sorted(os.listdir('/sys/class/net'))))"
        )
        response = client.post("/run", json={"code": code})
        assert response.status_code == 200
        data = response.json()
        assert data["exit_code"] == 0
        interfaces = set(json.loads(data["stdout"]))
        assert "eth0" in interfaces

    def test_network_disabled(self, client: TestClient) -> None:
        code = (
            "import json, os\nprint(json.dumps(sorted(os.listdir('/sys/class/net'))))"
        )
        response = client.post("/run", json={"code": code, "allow_network": False})
        assert response.status_code == 200
        data = response.json()
        assert data["exit_code"] == 0
        interfaces = set(json.loads(data["stdout"]))
        assert "eth0" not in interfaces

    def test_memory_limit(self, client: TestClient) -> None:
        code = "a = bytearray(200 * 1024 * 1024)"
        response = client.post("/run", json={"code": code, "mem_limit": "100m"})
        assert response.status_code == 200
        data = response.json()
        if data["exit_code"] == 0:
            pytest.skip("Docker does not enforce cgroup memory limits")
        assert data["exit_code"] == 137

    def test_execution_timeout(self, client: TestClient) -> None:
        code = "import time; time.sleep(10)"
        response = client.post("/run", json={"code": code, "timeout": 2})
        assert response.status_code == 200
        data = response.json()
        assert data["exit_code"] == -1


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for input validation."""

    def test_invalid_filename_rejected(self, client: TestClient) -> None:
        payload = {"files": {"../a.py": "x=1"}, "code": "print(1)"}
        response = client.post("/run", json=payload)
        assert response.status_code == 400
        assert "Invalid filename" in response.json()["detail"]

    def test_code_size_limit(self, client: TestClient) -> None:
        large_code = "a" * (1024 * 1024 + 1)
        response = client.post("/run", json={"code": large_code})
        assert response.status_code == 413

    def test_empty_code_without_entrypoint(self, client: TestClient) -> None:
        response = client.post("/run", json={"code": ""})
        assert response.status_code == 400

    def test_empty_code_with_entrypoint(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={"code": "", "entrypoint": "echo 'hello'", "image": "node"},
        )
        assert response.status_code == 200

    def test_reserved_env_var_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={
                "code": "print('test')",
                "environment": {"ENCODED_CODE": "malicious"},
            },
        )
        assert response.status_code == 422

    def test_unsupported_image_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/run", json={"code": "print('test')", "image": "unsupported"}
        )
        assert response.status_code == 404


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Tests for authentication."""

    def test_missing_api_key(self, client: TestClient) -> None:
        client.headers.pop("X-API-Key", None)
        response = client.post("/run", json={"code": "print('hello')"})
        assert response.status_code == 401

    def test_wrong_api_key(self, client: TestClient) -> None:
        client.headers["X-API-Key"] = "wrong-key"
        response = client.post("/run", json={"code": "print('hello')"})
        assert response.status_code == 403

    def test_second_valid_key(self, client: TestClient) -> None:
        client.headers["X-API-Key"] = TEST_API_KEYS.split(",")[1]
        response = client.post("/run", json={"code": "print('hello from second key')"})
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "hello from second key\n"


# =============================================================================
# Custom Entrypoint Tests
# =============================================================================


class TestCustomEntrypoint:
    """Tests for custom entrypoint functionality."""

    def test_python_custom_entrypoint(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={
                "code": "x = 42",
                "image": "python",
                "entrypoint": "python -c \"print('custom entrypoint')\"",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "custom entrypoint\n"

    def test_node_custom_entrypoint(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={
                "code": "",
                "image": "node",
                "entrypoint": "node -e \"console.log('inline node')\"",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "inline node\n"

    def test_entrypoint_with_files(self, client: TestClient) -> None:
        response = client.post(
            "/run",
            json={
                "code": "",
                "image": "node",
                "files": {"script.js": "console.log('script file')"},
                "entrypoint": "node script.js",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["stdout"] == "script file\n"


# =============================================================================
# Parallel Execution Tests
# =============================================================================


class TestParallelExecution:
    """Tests for concurrent execution."""

    def test_parallel_requests(self, client: TestClient) -> None:
        headers = {"X-API-Key": FIRST_TEST_KEY}
        payload1 = {"code": "import time; time.sleep(2); print('parallel 1')"}
        payload2 = {"code": "print('parallel 2')"}

        def make_request(payload: dict) -> object:
            return client.post("/run", json=payload, headers=headers)

        with ThreadPoolExecutor() as executor:
            start = time.time()
            f1 = executor.submit(make_request, payload1)
            f2 = executor.submit(make_request, payload2)
            r1, r2 = f1.result(), f2.result()
            elapsed = time.time() - start

        assert elapsed < 10
        assert r1.status_code == 200
        assert r1.json()["stdout"] == "parallel 1\n"
        assert r2.status_code == 200
        assert r2.json()["stdout"] == "parallel 2\n"


# =============================================================================
# Prebuilt Image Tests
# =============================================================================


class TestPrebuiltImages:
    """Tests for prebuilt image management."""

    def test_prebuild_image(self, client: TestClient) -> None:
        response = client.post(
            "/prebuild",
            json={
                "tag": "test-python-pkg",
                "base_image": "python",
                "setup_script": "pip install --no-cache-dir cowsay",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tag"] == "prebuilt-test-python-pkg"
        assert data["status"] == "ready"

    def test_run_with_prebuilt(self, client: TestClient) -> None:
        # Build first
        client.post(
            "/prebuild",
            json={
                "tag": "test-run-prebuilt",
                "base_image": "python",
                "setup_script": "pip install --no-cache-dir cowsay",
            },
        )
        # Then run
        response = client.post(
            "/run",
            json={
                "code": "import cowsay; print(cowsay.get_output_string('cow', 'prebuild works'))",
                "image": "prebuilt-test-run-prebuilt",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["exit_code"] == 0
        assert "prebuild works" in data["stdout"]

    def test_list_prebuilds(self, client: TestClient) -> None:
        client.post(
            "/prebuild",
            json={
                "tag": "test-list-item",
                "base_image": "python",
                "setup_script": "echo 'done'",
            },
        )
        response = client.get("/prebuilds")
        assert response.status_code == 200
        tags = [p["tag"] for p in response.json()["prebuilds"]]
        assert "prebuilt-test-list-item" in tags

    def test_delete_prebuild(self, client: TestClient) -> None:
        client.post(
            "/prebuild",
            json={
                "tag": "test-delete-me",
                "base_image": "python",
                "setup_script": "echo 'done'",
            },
        )
        response = client.delete("/prebuilds/test-delete-me")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # Verify it's gone
        response = client.get("/prebuilds/test-delete-me")
        assert response.status_code == 404

    def test_prebuild_failed_script(self, client: TestClient) -> None:
        response = client.post(
            "/prebuild",
            json={
                "tag": "test-fail-build",
                "base_image": "python",
                "setup_script": "pip install nonexistent-package-xyz-12345",
            },
        )
        assert response.status_code == 400
        assert "Build failed" in response.json()["detail"]

    def test_nonexistent_prebuild(self, client: TestClient) -> None:
        response = client.get("/prebuilds/nonexistent-xyz")
        assert response.status_code == 404

    def test_cleanup_old_prebuilds(self, client: TestClient) -> None:
        client.post(
            "/prebuild",
            json={
                "tag": "test-cleanup",
                "base_image": "python",
                "setup_script": "echo 'done'",
            },
        )
        time.sleep(2)
        response = client.post("/prebuilds/cleanup?max_age_days=0")
        assert response.status_code == 200

    def test_prebuild_invalid_base(self, client: TestClient) -> None:
        response = client.post(
            "/prebuild",
            json={
                "tag": "test-invalid",
                "base_image": "nonexistent",
                "setup_script": "echo hi",
            },
        )
        assert response.status_code == 422
