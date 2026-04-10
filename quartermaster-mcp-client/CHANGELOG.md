# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-10

### Added

- Initial release of `quartermaster-mcp-client`
- `McpClient` with async and sync context manager support
- SSE (Server-Sent Events) transport implementation
- Streamable HTTP transport implementation
- Transport auto-selection and factory pattern
- JSON-RPC 2.0 request/response handling
- Full tool discovery via `list_tools()`
- Tool invocation via `call_tool()`
- Resource listing via `list_resources()`
- Resource reading via `read_resource()`
- Server info retrieval via `server_info()`
- Custom exception hierarchy (`McpError`, `McpConnectionError`, `McpProtocolError`, etc.)
- Retry logic with exponential backoff and jitter
- Bearer token authentication support
- Custom HTTP header injection
- Comprehensive type hints (mypy strict compatible)
- PEP 561 `py.typed` marker
- 98 tests with 96% code coverage
- GitHub Actions CI/CD pipeline
- Pre-commit hooks configuration
