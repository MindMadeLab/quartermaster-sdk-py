#!/usr/bin/env python3
"""
MCP Stdio Client - Executes MCP server commands and calls tools via stdio transport

Environment variables:
- MCP_COMMAND: The command to start the MCP server (e.g., "uvx mcp-server-fetch")
- MCP_OPERATION: The operation to perform: "list_tools" or "call_tool" (default: "call_tool")
- MCP_TOOL_NAME: The name of the tool to call (required for call_tool operation)
- MCP_TOOL_ARGUMENTS: JSON string of arguments to pass to the tool
- MCP_ENV_*: Additional environment variables to pass to the MCP server (prefix stripped)
"""

import json
import os
import subprocess
import sys
import threading
from typing import Any

MCP_PROTOCOL_VERSION = "2024-11-05"
INITIALIZE_TIMEOUT = 60  # 60 seconds for slow package resolution
TOOL_CALL_TIMEOUT = 120


class McpStdioClient:
    def __init__(self, command: str, server_env: dict[str, str] | None = None):
        self.command = command
        self.server_env = server_env or {}
        self.process: subprocess.Popen | None = None
        self.message_id = 0
        self.pending_requests: dict[int, dict] = {}
        self.initialized = False
        self._lock = threading.Lock()
        self._response_event = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._exit_thread: threading.Thread | None = None
        self._last_response: dict | None = None
        self._server_stderr_lines: list[str] = []
        self._process_exited = False
        self._exit_code: int | None = None

    def start(self):
        env = os.environ.copy()
        env.update(self.server_env)

        for key in ["MCP_COMMAND", "MCP_TOOL_NAME", "MCP_TOOL_ARGUMENTS"]:
            env.pop(key, None)

        self.process = subprocess.Popen(
            self.command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )

        self._reader_thread = threading.Thread(target=self._read_responses, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        self._exit_thread = threading.Thread(target=self._monitor_exit, daemon=True)
        self._exit_thread.start()

    def _monitor_exit(self):
        if not self.process:
            return
        self._exit_code = self.process.wait()
        self._process_exited = True
        self._response_event.set()

    def _read_stderr(self):
        if not self.process or not self.process.stderr:
            return
        for line in self.process.stderr:
            line = line.rstrip("\n")
            self._server_stderr_lines.append(line)
            print(f"[MCP Server] {line}", file=sys.stderr)

    def _read_responses(self):
        if not self.process or not self.process.stdout:
            return

        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
                self._handle_message(message)
            except json.JSONDecodeError:
                print(f"[MCP Server stdout] {line}", file=sys.stderr)

    def _handle_message(self, message: dict):
        msg_id = message.get("id")
        if msg_id is not None and msg_id in self.pending_requests:
            with self._lock:
                self._last_response = message
                self._response_event.set()

    def _send_request(
        self, method: str, params: dict | None = None, timeout: int = 30
    ) -> Any:
        self.message_id += 1
        request_id = self.message_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        with self._lock:
            self.pending_requests[request_id] = {}
            self._response_event.clear()
            self._last_response = None

        if self._process_exited:
            raise RuntimeError(
                f"MCP server exited with code {self._exit_code} before {method} could be sent"
            )

        if self.process and self.process.stdin:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()

        if not self._response_event.wait(timeout):
            raise TimeoutError(f"{method} timed out after {timeout}s")

        if self._process_exited and self._last_response is None:
            stderr_tail = "\n".join(self._server_stderr_lines[-20:])
            raise RuntimeError(
                f"MCP server process exited with code {self._exit_code} during {method}.\n"
                f"Server output:\n{stderr_tail}"
            )

        with self._lock:
            response = self._last_response
            self.pending_requests.pop(request_id, None)

        if response and "error" in response:
            error = response["error"]
            raise Exception(error.get("message", str(error)))

        return response.get("result") if response else None

    def _send_notification(self, method: str, params: dict | None = None):
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }

        if self.process and self.process.stdin:
            self.process.stdin.write(json.dumps(notification) + "\n")
            self.process.stdin.flush()

    def initialize(self) -> dict:
        result = self._send_request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "quartermaster-code-runner",
                    "version": "1.0.0",
                },
            },
            timeout=INITIALIZE_TIMEOUT,
        )

        self._send_notification("notifications/initialized")
        self.initialized = True
        return result

    def list_tools(self) -> list[dict]:
        result = self._send_request("tools/list", {})
        return result.get("tools", []) if result else []

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        result = self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=TOOL_CALL_TIMEOUT,
        )

        content = result.get("content", []) if result else []
        text_parts = []

        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                text_parts.append(f"[Image: {item.get('mimeType', 'unknown')}]")
            elif item.get("type") == "resource":
                text_parts.append(f"[Resource: {item.get('uri', 'unknown')}]")

        return "\n".join(text_parts)

    def close(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def main():
    command = os.environ.get("MCP_COMMAND")
    operation = os.environ.get("MCP_OPERATION", "call_tool")
    tool_name = os.environ.get("MCP_TOOL_NAME")
    tool_arguments_json = os.environ.get("MCP_TOOL_ARGUMENTS", "{}")

    if not command:
        print(
            "Error: MCP_COMMAND environment variable is required",
            file=sys.stderr,
        )
        sys.exit(1)

    if operation == "call_tool" and not tool_name:
        print(
            "Error: MCP_TOOL_NAME environment variable is required for call_tool operation",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        tool_arguments = json.loads(tool_arguments_json)
    except json.JSONDecodeError:
        print("Error: MCP_TOOL_ARGUMENTS must be valid JSON", file=sys.stderr)
        sys.exit(1)

    server_env = {}
    for key, value in os.environ.items():
        if key.startswith("MCP_ENV_"):
            server_env[key[8:]] = value

    client = McpStdioClient(command, server_env)

    try:
        print(f"Starting MCP server: {command}", file=sys.stderr)
        client.start()

        print("Initializing MCP connection...", file=sys.stderr)
        client.initialize()

        if operation == "list_tools":
            print("Listing tools...", file=sys.stderr)
            tools = client.list_tools()
            print(json.dumps(tools))
        else:
            print(f"Calling tool: {tool_name}", file=sys.stderr)
            result = client.call_tool(tool_name or "", tool_arguments)
            print(result)

        client.close()
        sys.exit(0)
    except Exception as error:
        print(f"MCP Error: {error}", file=sys.stderr)
        if client._server_stderr_lines:
            print("--- MCP Server stderr ---", file=sys.stderr)
            for line in client._server_stderr_lines:
                print(line, file=sys.stderr)
            print("--- End server stderr ---", file=sys.stderr)
        client.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
