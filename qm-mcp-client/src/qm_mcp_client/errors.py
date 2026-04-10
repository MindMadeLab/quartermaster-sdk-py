"""Exception types for MCP client errors."""


class McpError(Exception):
    """Base exception for all MCP client errors."""

    pass


class McpConnectionError(McpError):
    """Raised when the connection to the MCP server fails."""

    pass


class McpProtocolError(McpError):
    """Raised when the MCP protocol is violated or response is malformed."""

    pass


class McpToolNotFoundError(McpError):
    """Raised when a requested tool is not available on the server."""

    pass


class McpTimeoutError(McpError):
    """Raised when an operation times out."""

    pass


class McpAuthenticationError(McpError):
    """Raised when authentication with the server fails."""

    pass


class McpServerError(McpError):
    """Raised when the server returns an error response."""

    def __init__(self, message: str, code: int | None = None) -> None:
        """Initialize server error.

        Args:
            message: Error message from server.
            code: JSON-RPC error code if available.
        """
        self.code = code
        super().__init__(message)
