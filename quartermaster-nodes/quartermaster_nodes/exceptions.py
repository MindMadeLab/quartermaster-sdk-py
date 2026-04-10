"""Exceptions used by quartermaster-nodes."""


class QMNodesError(Exception):
    """Base exception for all quartermaster-nodes errors."""


class MissingMemoryIDException(QMNodesError):
    """Raised when a required memory/thought ID is not set."""

    def __init__(self, message: str = "Memory ID is required but was not provided."):
        super().__init__(message)


class ProcessStopException(QMNodesError):
    """Raised to halt flow processing (e.g., awaiting user input)."""

    def __init__(self, message: str = "Process stopped."):
        super().__init__(message)


class NodeNotFoundError(QMNodesError):
    """Raised when a node type is not found in the registry."""

    def __init__(self, node_type: str, version: str = ""):
        msg = f"Node type '{node_type}' not found"
        if version:
            msg += f" with version '{version}'"
        super().__init__(msg)
        self.node_type = node_type
        self.version = version


class NodeExecutionError(QMNodesError):
    """Raised when a node fails during execution."""

    def __init__(self, node_name: str, original_error: Exception):
        super().__init__(f"Node '{node_name}' failed: {original_error}")
        self.node_name = node_name
        self.original_error = original_error


class ExpressionEvaluationError(QMNodesError):
    """Raised when an expression evaluation fails."""

    def __init__(self, expression: str, error: str):
        super().__init__(f"Failed to evaluate expression '{expression}': {error}")
        self.expression = expression
        self.error = error
