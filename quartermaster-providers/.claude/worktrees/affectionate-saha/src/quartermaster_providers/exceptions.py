"""Exceptions for quartermaster-providers.

All providers raise these consistent exceptions, making error handling
uniform regardless of which provider is being used.
"""


class ProviderError(Exception):
    """Base exception for all provider errors.

    Attributes:
        message: Human-readable error description.
        provider: Name of the provider that raised the error.
        status_code: HTTP status code if applicable.
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
    ):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(ProviderError):
    """Raised when API authentication fails (invalid/missing API key)."""

    def __init__(self, message: str = "Authentication failed", provider: str | None = None):
        super().__init__(message, provider=provider, status_code=401)


class RateLimitError(ProviderError):
    """Raised when the provider's rate limit is exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying, if provided by the API.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        provider: str | None = None,
        retry_after: float | None = None,
    ):
        self.retry_after = retry_after
        super().__init__(message, provider=provider, status_code=429)


class InvalidModelError(ProviderError):
    """Raised when the requested model is not available or doesn't exist."""

    def __init__(self, model: str, provider: str | None = None):
        self.model = model
        super().__init__(f"Model '{model}' is not available", provider=provider, status_code=404)


class InvalidRequestError(ProviderError):
    """Raised when the request is malformed or has invalid parameters."""

    def __init__(self, message: str = "Invalid request", provider: str | None = None):
        super().__init__(message, provider=provider, status_code=400)


class ContentFilterError(ProviderError):
    """Raised when content is blocked by the provider's safety filters."""

    def __init__(
        self, message: str = "Content blocked by safety filter", provider: str | None = None
    ):
        super().__init__(message, provider=provider, status_code=400)


class ContextLengthError(ProviderError):
    """Raised when the input exceeds the model's context window."""

    def __init__(self, message: str = "Context length exceeded", provider: str | None = None):
        super().__init__(message, provider=provider, status_code=400)


class ServiceUnavailableError(ProviderError):
    """Raised when the provider's service is temporarily unavailable."""

    def __init__(self, message: str = "Service unavailable", provider: str | None = None):
        super().__init__(message, provider=provider, status_code=503)
