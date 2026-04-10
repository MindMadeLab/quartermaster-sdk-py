"""Unit tests for security/authentication module."""

from __future__ import annotations

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from quartermaster_code_runner.security import configure_auth, verify_auth


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with auth dependency."""
    app = FastAPI()

    @app.get("/protected")
    async def protected(auth: str | None = Depends(verify_auth)) -> dict:
        return {"auth": auth}

    return app


class TestNoAuth:
    """Tests when authentication is disabled."""

    def test_no_auth_allows_all(self) -> None:
        configure_auth(api_keys=[], auth_token=None)
        client = TestClient(_make_app())
        resp = client.get("/protected")
        assert resp.status_code == 200
        assert resp.json()["auth"] is None

    def test_no_auth_ignores_bad_key(self) -> None:
        configure_auth(api_keys=[], auth_token=None)
        client = TestClient(_make_app())
        resp = client.get("/protected", headers={"X-API-Key": "bad"})
        assert resp.status_code == 200


class TestApiKeyAuth:
    """Tests for API key authentication."""

    def test_valid_api_key(self) -> None:
        configure_auth(api_keys=["key1", "key2"], auth_token=None)
        client = TestClient(_make_app())
        resp = client.get("/protected", headers={"X-API-Key": "key1"})
        assert resp.status_code == 200
        assert resp.json()["auth"] == "key1"

    def test_second_api_key(self) -> None:
        configure_auth(api_keys=["key1", "key2"], auth_token=None)
        client = TestClient(_make_app())
        resp = client.get("/protected", headers={"X-API-Key": "key2"})
        assert resp.status_code == 200

    def test_invalid_api_key(self) -> None:
        configure_auth(api_keys=["key1"], auth_token=None)
        client = TestClient(_make_app())
        resp = client.get("/protected", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 403
        assert "Invalid API Key" in resp.json()["detail"]

    def test_missing_api_key(self) -> None:
        configure_auth(api_keys=["key1"], auth_token=None)
        client = TestClient(_make_app())
        resp = client.get("/protected")
        assert resp.status_code == 401
        assert "Not authenticated" in resp.json()["detail"]


class TestBearerTokenAuth:
    """Tests for Bearer token authentication."""

    def test_valid_bearer_token(self) -> None:
        configure_auth(api_keys=[], auth_token="my-token")
        client = TestClient(_make_app())
        resp = client.get(
            "/protected",
            headers={"Authorization": "Bearer my-token"},
        )
        assert resp.status_code == 200
        assert resp.json()["auth"] == "my-token"

    def test_invalid_bearer_token(self) -> None:
        configure_auth(api_keys=[], auth_token="my-token")
        client = TestClient(_make_app())
        resp = client.get(
            "/protected",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_missing_bearer_prefix(self) -> None:
        configure_auth(api_keys=[], auth_token="my-token")
        client = TestClient(_make_app())
        resp = client.get(
            "/protected",
            headers={"Authorization": "my-token"},
        )
        assert resp.status_code == 401


class TestMixedAuth:
    """Tests with both API keys and Bearer token configured."""

    def test_api_key_works(self) -> None:
        configure_auth(api_keys=["key1"], auth_token="token")
        client = TestClient(_make_app())
        resp = client.get("/protected", headers={"X-API-Key": "key1"})
        assert resp.status_code == 200

    def test_bearer_works(self) -> None:
        configure_auth(api_keys=["key1"], auth_token="token")
        client = TestClient(_make_app())
        resp = client.get(
            "/protected",
            headers={"Authorization": "Bearer token"},
        )
        assert resp.status_code == 200

    def test_neither_fails(self) -> None:
        configure_auth(api_keys=["key1"], auth_token="token")
        client = TestClient(_make_app())
        resp = client.get("/protected")
        assert resp.status_code == 401
