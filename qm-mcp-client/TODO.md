# qm-mcp-client — Extraction TODO

A lightweight, production-ready MCP Protocol client library extracted from the Quartermaster backend.

## Overview

This library provides a zero-dependency (except httpx) Python client for interacting with Model Context Protocol (MCP) servers. It supports both async and sync interfaces with multiple transport layers (SSE and Streamable HTTP).

## Phase 1: Core Extraction (Core Functionality)

### 1.1 Extract Client Logic
- [x] Implement core MCP client with JSON-RPC 2.0 request/response handling
- [x] Implement SSE parsing logic: `parse_sse_response()`
- [x] Implement JSON Schema type parsing: `parse_json_schema_type()`
- [x] Implement parameter parsing: `parse_tool_parameters()` and related helpers
- [x] Implement tool discovery and calling logic
- [x] Use standalone dataclasses (ToolParameter, ToolParameterOption) instead of Django models

### 1.2 Implement Async/Sync Dual Interface
- [x] Implement async methods: `server_info()`, `list_tools()`, `call_tool()`, `list_resources()`, `read_resource()`
- [x] Implement sync wrappers using asyncio.run() with event loop detection
- [x] Support both `async with` and `with` context managers
- [x] Sync wrapper uses ThreadPoolExecutor when event loop is already running

### 1.3 Error Handling
- [x] Custom exception hierarchy: McpError, McpConnectionError, McpProtocolError, McpToolNotFoundError, McpTimeoutError, McpAuthenticationError, McpServerError
- [x] Map JSON-RPC error codes to exceptions
- [x] Helpful error messages with context

### 1.4 Remove Framework Dependencies
- [x] Zero framework dependencies — only httpx as external dependency
- [x] Standard json module, typing module, dataclasses throughout

## Phase 2: Transport Layers

### 2.1 SSE Transport (Server-Sent Events)
- [x] Implement SSETransport class in `transports.py`
- [x] Handle streaming HTTP responses
- [x] Parse multipart SSE frames ("data: {...}\n\n")
- [x] Support server-sent stream results
- [x] Handle plain JSON fallback

### 2.2 Streamable HTTP Transport
- [x] Implement StreamableTransport class
- [x] Handle chunked Transfer-Encoding
- [x] Parse JSON responses
- [x] Handle stream timeouts

### 2.3 Transport Selection
- [x] Implement transport factory (`create_transport()`)
- [x] Support "sse" and "streamable" transport types

## Phase 3: Quality & Testing

### 3.1 Unit Tests
- [x] Test SSE parsing: `test_parse_sse_response()` (7 tests)
- [x] Test schema type extraction: `test_parse_json_schema_type()` (14 tests)
- [x] Test parameter parsing: `test_parse_tool_parameters()` (12 tests)
- [x] Test client initialization (12 tests)
- [x] Test context managers (async and sync) (4 tests)
- [x] Test sync/async wrapper switching (5 tests)

### 3.2 Integration Tests
- [x] Test with mock MCP server (respx mocking)
- [x] Test tool listing and invocation
- [x] Test resource reading
- [x] Test server info retrieval
- [x] Test auth header injection (2 tests)

### 3.3 Error Handling Tests
- [x] Network failures (connection refused, timeout)
- [x] Malformed responses (invalid JSON, missing fields)
- [x] Protocol violations (invalid JSON-RPC version)
- [x] Tool not found errors
- [x] Server error responses

### 3.4 Performance Tests
- [x] Retry logic under transient failures (4 tests)
- [x] Timeout configuration

### 3.5 Type Safety
- [x] Comprehensive type hints throughout
- [x] mypy --strict passes with zero errors
- [x] py.typed PEP 561 marker added

## Phase 4: Documentation

### 4.1 Code Documentation
- [x] Docstrings for all public classes and methods (Google style)
- [x] Type hint stubs via py.typed

### 4.2 User Documentation
- [x] README.md with quick start
- [x] CHANGELOG.md

### 4.3 Contributor Guide
- [x] CONTRIBUTING.md with development setup
- [x] Testing guidelines
- [x] Code style expectations (ruff, mypy)

## Phase 5: CI/CD & Release

### 5.1 GitHub Actions Workflow
- [x] Lint (ruff check + format)
- [x] Type checking (mypy --strict)
- [x] Unit tests (pytest) across Python 3.11-3.13
- [x] Code coverage (pytest-cov, 85% threshold)
- [x] Build wheel and sdist

### 5.2 Pre-Commit Hooks
- [x] Configure `.pre-commit-config.yaml`
- [x] Ruff lint + format
- [x] mypy for type checking
- [x] Trailing whitespace, EOF fixers

### 5.3 PyPI Publication
- [x] GitHub Actions publish workflow on release tags
- [ ] Test publication to Test PyPI first
- [ ] Create release on GitHub

## Phase 6: Advanced Features (Future)

### 6.1 Connection Management
- [ ] Connection pooling configuration
- [ ] Keep-alive settings
- [ ] Max connections limit

### 6.2 Retry Strategy
- [x] Exponential backoff with jitter
- [x] Configurable max retries
- [ ] Idempotency key handling

### 6.3 Authentication
- [x] Token-based auth (Authorization header)
- [x] Custom header injection
- [ ] OAuth2 support (optional)

### 6.4 Monitoring & Debugging
- [x] Structured logging
- [ ] Request/response debugging mode
- [ ] Timing instrumentation

## Results Summary

- **98 tests** passing
- **96.5% code coverage** (above 85% threshold)
- **mypy --strict** clean
- **ruff** lint and format clean
- **Zero external dependencies** except httpx
