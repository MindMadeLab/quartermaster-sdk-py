"""Tests for exception hierarchy."""

import pytest

from qm_providers.exceptions import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidModelError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
)


class TestProviderError:
    def test_basic(self):
        e = ProviderError("Something failed")
        assert str(e) == "Something failed"
        assert e.provider is None
        assert e.status_code is None

    def test_with_provider(self):
        e = ProviderError("Failed", provider="openai", status_code=500)
        assert e.provider == "openai"
        assert e.status_code == 500

    def test_is_exception(self):
        assert issubclass(ProviderError, Exception)


class TestAuthenticationError:
    def test_default_message(self):
        e = AuthenticationError()
        assert "Authentication failed" in str(e)
        assert e.status_code == 401

    def test_custom_message(self):
        e = AuthenticationError("Bad key", provider="anthropic")
        assert str(e) == "Bad key"
        assert e.provider == "anthropic"

    def test_inheritance(self):
        e = AuthenticationError()
        assert isinstance(e, ProviderError)


class TestRateLimitError:
    def test_default(self):
        e = RateLimitError()
        assert e.status_code == 429
        assert e.retry_after is None

    def test_with_retry(self):
        e = RateLimitError(retry_after=30.0)
        assert e.retry_after == 30.0

    def test_inheritance(self):
        assert issubclass(RateLimitError, ProviderError)


class TestInvalidModelError:
    def test_basic(self):
        e = InvalidModelError("gpt-99")
        assert "gpt-99" in str(e)
        assert e.model == "gpt-99"
        assert e.status_code == 404

    def test_inheritance(self):
        assert issubclass(InvalidModelError, ProviderError)


class TestInvalidRequestError:
    def test_basic(self):
        e = InvalidRequestError("Bad params")
        assert str(e) == "Bad params"
        assert e.status_code == 400


class TestContentFilterError:
    def test_basic(self):
        e = ContentFilterError()
        assert "safety filter" in str(e).lower() or "blocked" in str(e).lower()
        assert e.status_code == 400


class TestContextLengthError:
    def test_basic(self):
        e = ContextLengthError()
        assert e.status_code == 400


class TestServiceUnavailableError:
    def test_basic(self):
        e = ServiceUnavailableError()
        assert e.status_code == 503


class TestExceptionHierarchy:
    """Verify all exceptions can be caught as ProviderError."""

    def test_catch_all(self):
        exceptions = [
            AuthenticationError(),
            RateLimitError(),
            InvalidModelError("x"),
            InvalidRequestError(),
            ContentFilterError(),
            ContextLengthError(),
            ServiceUnavailableError(),
        ]
        for exc in exceptions:
            with pytest.raises(ProviderError):
                raise exc
