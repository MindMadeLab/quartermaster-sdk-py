"""Use the built-in tools: ReadFileTool, WriteFileTool, WebRequestTool.

Demonstrates the ready-made tools that ship with quartermaster-tools for file I/O
and HTTP requests.  Each tool follows the same AbstractTool interface.
"""

from __future__ import annotations

import os
import tempfile

try:
    from quartermaster_tools.builtin.file_read import ReadFileTool
    from quartermaster_tools.builtin.file_write import WriteFileTool
except ImportError:
    raise SystemExit("Install quartermaster-tools first:  pip install -e quartermaster-tools")


def main() -> None:
    # Use a temporary directory so the example is self-contained
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "hello.txt")

        # -- WriteFileTool ----------------------------------------------------
        writer = WriteFileTool(allowed_base_dir=tmpdir, create_dirs=True)
        print(f"WriteFileTool name   : {writer.name()}")
        print(f"WriteFileTool version: {writer.version()}")

        result = writer.run(path=filepath, content="Hello from Quartermaster!\n")
        print(f"Write result: success={result.success}, bytes={result.data.get('bytes_written')}")

        # Append mode
        result = writer.run(path=filepath, content="Second line.\n", append=True)
        print(f"Append result: success={result.success}")

        # -- ReadFileTool -----------------------------------------------------
        reader = ReadFileTool(allowed_base_dir=tmpdir)
        print(f"\nReadFileTool name   : {reader.name()}")
        print(f"ReadFileTool version: {reader.version()}")

        result = reader.run(path=filepath)
        print(f"Read result: success={result.success}")
        print(f"Content:\n{result.data.get('content', '')}")

        # Security: reading outside allowed_base_dir fails
        result = reader.run(path="/tmp/outside.txt")
        print(f"Outside base dir: success={result.success}, error={result.error}")

        # -- WebRequestTool (optional, needs httpx) ---------------------------
        try:
            from quartermaster_tools.builtin.web_request import WebRequestTool

            web = WebRequestTool(timeout=10)
            print(f"\nWebRequestTool name   : {web.name()}")
            print(f"WebRequestTool version: {web.version()}")
            print("Parameters:")
            for p in web.parameters():
                print(f"  {p.name:10s} ({p.type}) required={p.required}")
            # Skipping actual HTTP call to keep the example offline-friendly
            print("(Skipping live HTTP request -- pass a URL to try it yourself)")
        except ImportError:
            print("\nhttpx not installed -- skipping WebRequestTool demo.")
            print("Install with:  pip install quartermaster-tools[web]")


if __name__ == "__main__":
    main()
