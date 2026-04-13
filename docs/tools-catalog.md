# Quartermaster Tools Catalog

Comprehensive reference for all built-in tools in the `quartermaster-tools` package. This catalog covers 50+ tools organized across 14 categories, with parameter specifications, usage examples, and extension patterns.

---

## Table of Contents

1. [Tool System Overview](#tool-system-overview)
2. [Code Execution](#code-execution)
3. [Data Processing](#data-processing)
4. [File I/O](#file-io)
5. [Filesystem](#filesystem)
6. [Database](#database)
7. [Web Search](#web-search)
8. [Web Requests](#web-requests)
9. [Memory](#memory)
10. [Vector DB](#vector-db)
11. [Email](#email)
12. [Messaging](#messaging)
13. [Browser Automation](#browser-automation)
14. [Observability](#observability)
15. [Privacy / PII Detection](#privacy--pii-detection)
16. [Compliance (EU AI Act)](#compliance-eu-ai-act)
17. [A2A (Agent-to-Agent Protocol)](#a2a-agent-to-agent-protocol)
18. [Agent Sessions](#agent-sessions)
19. [Creating Custom Tools with @tool](#creating-custom-tools-with-tool)
20. [Registering Custom Tools](#registering-custom-tools)
21. [Bridging Tools to LLM Providers](#bridging-tools-to-llm-providers)

---

## Tool System Overview

Every tool in Quartermaster extends `AbstractTool` and implements:

- `name()` -- unique identifier string
- `version()` -- semver string
- `parameters()` -- list of `ToolParameter` definitions
- `info()` -- `ToolDescriptor` with metadata
- `run(**kwargs)` -- execute the tool and return a `ToolResult`

`ToolResult` is a standardized return type with fields:

| Field     | Type            | Description                        |
|-----------|-----------------|------------------------------------|
| `success` | `bool`          | Whether the operation succeeded    |
| `data`    | `dict` or None  | Result payload on success          |
| `error`   | `str` or None   | Error message on failure           |
| `metadata`| `dict` or None  | Optional extra metadata            |

Tools that run subprocesses extend `AbstractLocalTool`, which adds timeout enforcement and `prepare_command()` for subprocess construction.

### Installation extras

Most built-in tools have zero dependencies. Optional features require extras:

```bash
pip install quartermaster-tools          # Core tools (code, data, filesystem, memory, database)
pip install quartermaster-tools[web]     # Web search, web request, scraper, messaging (httpx)
pip install quartermaster-tools[browser] # Browser automation (playwright)
pip install quartermaster-tools[privacy] # PII detection (included by default, no extra deps)
```

---

## Code Execution

Tools for running code in sandboxed subprocesses.

| Tool | Class | Description |
|------|-------|-------------|
| `python_executor` | `PythonExecutorTool` | Execute Python code via `python3 -c` subprocess |
| `javascript_executor` | `JavaScriptExecutorTool` | Execute JavaScript code via `node -e` subprocess |
| `shell_executor` | `ShellExecutorTool` | Execute shell commands via `sh -c` subprocess |
| `eval_math` | `EvalMathTool` | Safely evaluate math expressions using AST parsing |

### python_executor

Execute Python code in a subprocess with timeout enforcement.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `code` | string | yes | -- | Python source code to execute |
| `timeout` | number | no | 30 | Maximum execution time in seconds |

```python
from quartermaster_tools import PythonExecutorTool

tool = PythonExecutorTool(timeout=30)
result = tool.run(code="print(2 + 2)")
# result.data == {"stdout": "4\n", "stderr": "", "exit_code": 0}
```

### javascript_executor

Execute JavaScript code via Node.js. Gracefully handles missing Node.js.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `code` | string | yes | -- | JavaScript source code to execute |
| `timeout` | number | no | 30 | Maximum execution time in seconds |

```python
from quartermaster_tools import JavaScriptExecutorTool

tool = JavaScriptExecutorTool()
result = tool.run(code="console.log(JSON.stringify({a: 1}))")
```

### shell_executor

Execute shell commands with a blocked-command safety list preventing destructive operations like `rm -rf /`, `mkfs`, and fork bombs.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `command` | string | yes | -- | Shell command to execute |
| `timeout` | number | no | 30 | Maximum execution time in seconds |
| `working_dir` | string | no | None | Working directory for execution |

```python
from quartermaster_tools import ShellExecutorTool

tool = ShellExecutorTool(timeout=10)
result = tool.run(command="ls -la /tmp")
```

### eval_math

Safe mathematical expression evaluation using AST parsing. Never calls `eval()` or `exec()`. Supports arithmetic (`+`, `-`, `*`, `/`, `//`, `%`, `**`), comparisons, and functions (`abs`, `round`, `min`, `max`, `sqrt`, `int`, `float`), plus constants (`pi`, `e`, `inf`).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `expression` | string | yes | -- | Mathematical expression to evaluate |

```python
from quartermaster_tools import EvalMathTool

tool = EvalMathTool()
result = tool.run(expression="sqrt(16) + pi")
# result.data == {"result": 7.141592653589793, "expression": "sqrt(16) + pi"}
```

---

## Data Processing

Tools for parsing, converting, and filtering structured data.

| Tool | Class | Description |
|------|-------|-------------|
| `parse_csv` | `ParseCSVTool` | Parse CSV data from file or string |
| `parse_json` | `ParseJSONTool` | Parse JSON data with optional JMESPath query |
| `parse_yaml` | `ParseYAMLTool` | Parse YAML data from file or string |
| `parse_xml` | `ParseXMLTool` | Parse XML data with optional XPath query |
| `convert_format` | `ConvertFormatTool` | Convert between CSV, JSON, and YAML |
| `data_filter` | `DataFilterTool` | Filter, sort, and limit structured data |

### parse_csv

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | string | yes | -- | File path or raw CSV string |
| `delimiter` | string | no | `,` | Column delimiter character |
| `has_headers` | boolean | no | true | Whether first row contains headers |

```python
from quartermaster_tools import ParseCSVTool

tool = ParseCSVTool()
result = tool.run(source="name,age\nAlice,30\nBob,25")
# result.data["rows"] == [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]
```

### parse_json

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | string | yes | -- | File path or raw JSON string |
| `query` | string | no | None | JMESPath query to apply to parsed data |

```python
from quartermaster_tools import ParseJSONTool

tool = ParseJSONTool()
result = tool.run(source='{"users": [{"name": "Alice"}]}', query="users[0].name")
# result.data["result"] == "Alice"
```

### parse_yaml

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | string | yes | -- | File path or raw YAML string |

Requires `pyyaml`. Install with `pip install pyyaml`.

### parse_xml

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | string | yes | -- | File path or raw XML string |
| `xpath` | string | no | None | XPath expression to select elements |

### convert_format

Convert data between CSV, JSON, and YAML formats.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | string | yes | -- | Data string or file path |
| `from_format` | string | yes | -- | Source format: `csv`, `json`, or `yaml` |
| `to_format` | string | yes | -- | Target format: `csv`, `json`, or `yaml` |

```python
from quartermaster_tools import ConvertFormatTool

tool = ConvertFormatTool()
result = tool.run(source='[{"a": 1}]', from_format="json", to_format="csv")
# result.data["output"] contains CSV string
```

### data_filter

Filter, sort, and limit lists of dicts using safe Python expressions.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data` | array | yes | -- | List of dicts to process |
| `filter_expression` | string | no | None | Python expression with `row` variable (e.g. `row['age'] > 18`) |
| `sort_by` | string | no | None | Key name to sort by |
| `limit` | number | no | None | Maximum rows to return |

```python
from quartermaster_tools import DataFilterTool

tool = DataFilterTool()
result = tool.run(
    data=[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 17}],
    filter_expression="row['age'] >= 18",
    sort_by="name",
)
# result.data["rows"] == [{"name": "Alice", "age": 30}]
```

---

## File I/O

Tools for reading and writing file content with security validation.

| Tool | Class | Description |
|------|-------|-------------|
| `read_file` | `ReadFileTool` | Read file content with path validation and size limits |
| `write_file` | `WriteFileTool` | Write content to file with security checks |

### read_file

Reads text files with security features: path validation, blocked system paths (`/etc/`, `/proc/`, `/sys/`, `/dev/`), configurable max file size (default 10 MB), optional base directory restriction, and encoding allowlist.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | yes | -- | Path to the file to read |
| `encoding` | string | no | `utf-8` | Text encoding |

```python
from quartermaster_tools import ReadFileTool

tool = ReadFileTool(max_file_size=5*1024*1024, allowed_base_dir="/app/data")
result = tool.run(path="/app/data/config.json")
# result.data["content"] contains the file text
```

### write_file

Writes text to files with security features: blocked system paths, max content size (default 10 MB), append mode support, and optional auto-creation of parent directories.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | yes | -- | Path to the file to write |
| `content` | string | yes | -- | Text content to write |
| `encoding` | string | no | `utf-8` | Text encoding |
| `append` | boolean | no | false | Append instead of overwrite |

```python
from quartermaster_tools import WriteFileTool

tool = WriteFileTool(create_dirs=True)
result = tool.run(path="/tmp/output.txt", content="Hello, world!")
# result.data == {"path": "/tmp/output.txt", "bytes_written": 13, "mode": "overwrite"}
```

---

## Filesystem

Tools for filesystem operations with path validation and security.

| Tool | Class | Description |
|------|-------|-------------|
| `list_directory` | `ListDirectoryTool` | List directory entries with metadata |
| `find_files` | `FindFilesTool` | Find files using glob patterns and optional regex |
| `grep` | `GrepTool` | Search file contents for regex patterns |
| `file_info` | `FileInfoTool` | Get file metadata (size, type, permissions, MIME type) |
| `copy_file` | `CopyFileTool` | Copy a file or directory |
| `move_file` | `MoveFileTool` | Move or rename a file or directory |
| `delete_file` | `DeleteFileTool` | Delete a file or directory (requires confirmation) |
| `create_directory` | `CreateDirectoryTool` | Create a directory with optional parent creation |

All filesystem tools accept an optional `allowed_base_dir` constructor parameter to restrict operations to a specific directory tree.

### list_directory

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | yes | -- | Directory path to list |
| `recursive` | boolean | no | false | Recurse into subdirectories |
| `pattern` | string | no | `*` | Glob pattern to filter entries |
| `include_hidden` | boolean | no | false | Include dot-prefixed files |

### find_files

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `root_path` | string | yes | -- | Root directory to search from |
| `pattern` | string | yes | -- | Glob pattern (e.g. `**/*.py`) |
| `name_pattern` | string | no | None | Regex to further filter file names |

### grep

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | yes | -- | File or directory to search |
| `pattern` | string | yes | -- | Regex pattern to search for |
| `recursive` | boolean | no | true | Recurse into subdirectories |
| `context_lines` | number | no | 0 | Context lines around matches |

### file_info

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | yes | -- | Path to inspect |

Returns: size, modified time, created time, type, permissions, and MIME type.

### copy_file

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | string | yes | -- | Source path |
| `destination` | string | yes | -- | Destination path |

### move_file

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | string | yes | -- | Source path |
| `destination` | string | yes | -- | Destination path |

### delete_file

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | yes | -- | Path to delete |
| `confirm` | boolean | yes | -- | Must be `true` to proceed |

### create_directory

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | yes | -- | Directory path to create |
| `parents` | boolean | no | true | Create parent directories as needed |

---

## Database

SQLite database tools using parameterized queries to prevent SQL injection.

| Tool | Class | Description |
|------|-------|-------------|
| `sqlite_query` | `SQLiteQueryTool` | Execute read-only SQL queries |
| `sqlite_write` | `SQLiteWriteTool` | Execute write SQL statements (requires confirmation) |
| `sqlite_schema` | `SQLiteSchemaTool` | Introspect database schema |

### sqlite_query

Executes SELECT queries only. Write statements are rejected.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `database` | string | yes | -- | Path to SQLite database file |
| `sql` | string | yes | -- | SQL SELECT query |
| `params` | array | no | None | Bind parameters for `?` placeholders |
| `max_rows` | number | no | 100 | Maximum rows to return |

```python
from quartermaster_tools import SQLiteQueryTool

tool = SQLiteQueryTool()
result = tool.run(database="app.db", sql="SELECT * FROM users WHERE age > ?", params=[18])
# result.data["rows"] is a list of dicts with column names as keys
```

### sqlite_write

Executes INSERT, UPDATE, DELETE, CREATE, DROP, ALTER statements. Requires `confirm=True` as a safety guard.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `database` | string | yes | -- | Path to SQLite database file |
| `sql` | string | yes | -- | SQL write statement |
| `params` | array | no | None | Bind parameters |
| `confirm` | boolean | no | false | Must be `true` to execute |

### sqlite_schema

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `database` | string | yes | -- | Path to SQLite database file |
| `table` | string | no | None | Table name to inspect (omit to list all tables) |

---

## Web Search

Multiple search providers, from zero-config to API-based.

| Tool | Class | Description | API Key Required |
|------|-------|-------------|-----------------|
| `duckduckgo_search` | `DuckDuckGoSearchTool` | Search via DuckDuckGo HTML endpoint | No |
| `google_search` | `GoogleSearchTool` | Search via Google Custom Search API | Yes (`GOOGLE_API_KEY`, `GOOGLE_CSE_ID`) |
| `brave_search` | `BraveSearchTool` | Search via Brave Search API | Yes (`BRAVE_API_KEY`) |
| `json_api` | `JsonApiTool` | Call any JSON API with optional JMESPath filtering | Depends on API |
| `web_scraper` | `WebScraperTool` | Fetch and convert web pages to text/markdown/HTML | No |

### duckduckgo_search

Zero-configuration web search. No API key needed.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Search query string |
| `max_results` | number | no | 5 | Maximum results (max 20) |

```python
from quartermaster_tools import DuckDuckGoSearchTool

tool = DuckDuckGoSearchTool()
result = tool.run(query="python async patterns")
# result.data["results"] == [{"title": "...", "url": "...", "snippet": "..."}, ...]
```

### google_search

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Search query string |
| `num_results` | number | no | 5 | Number of results (1-10) |
| `language` | string | no | None | Language code (e.g. `en`, `de`) |
| `region` | string | no | None | Country code (e.g. `us`, `uk`) |

Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` environment variables.

### brave_search

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Search query string |
| `count` | number | no | 5 | Number of results (1-20) |
| `country` | string | no | None | Country code filter (e.g. `US`, `GB`) |
| `freshness` | string | no | None | Freshness filter: `day`, `week`, or `month` |

Requires `BRAVE_API_KEY` environment variable.

### json_api

Call any JSON API endpoint with auto-parsing and optional JMESPath filtering.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | yes | -- | API endpoint URL |
| `method` | string | no | `GET` | HTTP method: GET, POST, PUT, DELETE, PATCH |
| `headers` | object | no | None | HTTP headers |
| `body` | object | no | None | Request body (auto-serialized as JSON if dict/list) |
| `jmespath_filter` | string | no | None | JMESPath expression to filter response |

```python
from quartermaster_tools import JsonApiTool

tool = JsonApiTool()
result = tool.run(
    url="https://api.github.com/repos/octocat/hello-world",
    jmespath_filter="full_name",
)
```

### web_scraper

Fetch a URL and return content as plain text, markdown, or raw HTML.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | yes | -- | URL to scrape |
| `output_format` | string | no | `text` | Output: `text`, `markdown`, or `html` |
| `timeout` | number | no | 30 | Request timeout in seconds (max 120) |

---

## Web Requests

General-purpose HTTP client with SSRF protection.

| Tool | Class | Description |
|------|-------|-------------|
| `web_request` | `WebRequestTool` | Make HTTP requests (GET, POST, PUT, DELETE, PATCH) |

### web_request

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | yes | -- | Target URL |
| `method` | string | no | `GET` | HTTP method |
| `headers` | object | no | None | Request headers |
| `body` | string | no | None | Request body for POST/PUT/PATCH |

Security features:
- SSRF protection blocks requests to private/loopback/link-local IPs (127.x, 10.x, 172.16.x, 192.168.x, 169.254.x, etc.)
- Configurable timeout (default 30s, max 300s)
- Max response size (default 5 MB)
- Only `http` and `https` schemes allowed

```python
from quartermaster_tools import WebRequestTool

tool = WebRequestTool(timeout=15)
result = tool.run(url="https://httpbin.org/get", method="GET")
# result.data == {"body": "...", "status_code": 200, "headers": {...}, "url": "..."}
```

---

## Memory

In-memory key-value store for persisting variables across tool calls within a session.

| Tool | Class | Description |
|------|-------|-------------|
| `set_variable` | `SetVariableTool` | Store a key-value pair in memory |
| `get_variable` | `GetVariableTool` | Retrieve a variable from memory |
| `list_variables` | `ListVariablesTool` | List stored variable names |

### set_variable

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | yes | -- | Variable name (key) |
| `value` | string | yes | -- | Value to store |

### get_variable

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | yes | -- | Variable name to retrieve |
| `default` | string | no | None | Default value if not found |

### list_variables

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prefix` | string | no | None | Filter variable names by prefix |

```python
from quartermaster_tools import SetVariableTool, GetVariableTool

store = {}
SetVariableTool(store=store).run(name="user_id", value="abc123")
result = GetVariableTool(store=store).run(name="user_id")
# result.data["value"] == "abc123"
```

---

## Vector DB

Built-in vector storage, embedding, search, and document indexing. Zero external dependencies by default (uses hash-based embeddings). Optionally uses `sentence-transformers` for real semantic embeddings.

| Tool | Class | Description |
|------|-------|-------------|
| `embed_text` | `EmbedTextTool` | Generate vector embeddings for text |
| `vector_store` | `VectorStoreTool` | Store text with embeddings in a collection |
| `vector_search` | `VectorSearchTool` | Search documents by cosine similarity |
| `hybrid_search` | `HybridSearchTool` | Combined semantic + keyword search |
| `document_index` | `DocumentIndexTool` | Chunk and index a document for search |

### embed_text

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | yes | -- | Text to embed |
| `model` | string | no | `builtin` | `builtin` for hash-based, or a sentence-transformers model name |
| `dimensions` | number | no | 384 | Embedding dimensions (builtin model only) |

### vector_store

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `collection` | string | yes | -- | Collection name |
| `text` | string | yes | -- | Text content to store |
| `metadata` | object | no | None | Metadata dict to attach |
| `embedding` | array | no | None | Pre-computed embedding (auto-generated if omitted) |
| `store_path` | string | no | None | JSON file for persistence (in-memory if omitted) |

### vector_search

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `collection` | string | yes | -- | Collection to search |
| `query` | string | yes | -- | Query text |
| `top_k` | number | no | 5 | Max results |
| `threshold` | number | no | 0.0 | Minimum similarity score |
| `store_path` | string | no | None | JSON store file path |

### hybrid_search

Combines cosine-similarity vector search with TF-based keyword scoring.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `collection` | string | yes | -- | Collection to search |
| `query` | string | yes | -- | Query text |
| `top_k` | number | no | 5 | Max results |
| `keyword_weight` | number | no | 0.3 | Weight for keyword score (0.0-1.0). Semantic weight = 1 - keyword_weight |
| `store_path` | string | no | None | JSON store file path |

### document_index

Reads a text file, splits it into overlapping chunks, embeds each, and stores in a vector collection.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | yes | -- | Path to text file |
| `collection` | string | yes | -- | Collection name |
| `chunk_size` | number | no | 500 | Characters per chunk |
| `chunk_overlap` | number | no | 50 | Overlap between chunks |
| `store_path` | string | no | None | JSON store for persistence |

```python
from quartermaster_tools.builtin.vector.index import DocumentIndexTool
from quartermaster_tools.builtin.vector.search import VectorSearchTool

# Index a document
DocumentIndexTool().run(file_path="docs/guide.txt", collection="guides")

# Search it
result = VectorSearchTool().run(collection="guides", query="how to configure")
# result.data["results"] == [{"text": "...", "metadata": {...}, "score": 0.85}, ...]
```

---

## Email

Send, read, and search emails via SMTP and IMAP. Uses only stdlib modules. Connection parameters fall back to environment variables.

| Tool | Class | Description |
|------|-------|-------------|
| `send_email` | `SendEmailTool` | Send email via SMTP (rate limited: 10/min) |
| `read_email` | `ReadEmailTool` | Read emails from IMAP mailbox |
| `search_email` | `SearchEmailTool` | Search emails via IMAP SEARCH |

### send_email

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `to` | string | yes | -- | Recipient email address |
| `subject` | string | yes | -- | Email subject line |
| `body` | string | yes | -- | Email body text |
| `cc` | string | no | None | CC recipients (comma-separated) |
| `bcc` | string | no | None | BCC recipients (comma-separated) |
| `smtp_host` | string | no | env `SMTP_HOST` | SMTP server host |
| `smtp_port` | number | no | 587 | SMTP server port |
| `smtp_user` | string | no | env `SMTP_USER` | SMTP username |
| `smtp_password` | string | no | env `SMTP_PASSWORD` | SMTP password |

### read_email

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `folder` | string | no | `INBOX` | Mailbox folder |
| `count` | number | no | 10 | Number of emails to fetch |
| `unread_only` | boolean | no | true | Only fetch unread emails |
| `imap_host` | string | no | env `IMAP_HOST` | IMAP server host |
| `imap_user` | string | no | env `IMAP_USER` | IMAP username |
| `imap_password` | string | no | env `IMAP_PASSWORD` | IMAP password |

### search_email

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Search text (subject and body) |
| `folder` | string | no | `INBOX` | Mailbox folder |
| `date_from` | string | no | None | From date (`DD-Mon-YYYY`) |
| `date_to` | string | no | None | To date (`DD-Mon-YYYY`) |
| `sender` | string | no | None | Filter by sender address |
| `imap_host` | string | no | env `IMAP_HOST` | IMAP server host |
| `imap_user` | string | no | env `IMAP_USER` | IMAP username |
| `imap_password` | string | no | env `IMAP_PASSWORD` | IMAP password |

---

## Messaging

Send messages to Slack, Discord, and generic webhooks. Requires `httpx`.

| Tool | Class | Description |
|------|-------|-------------|
| `slack_message` | `SlackMessageTool` | Send a message to a Slack channel |
| `slack_read` | `SlackReadTool` | Read messages from a Slack channel |
| `discord_message` | `DiscordMessageTool` | Send a message to Discord via webhook |
| `webhook_notify` | `WebhookNotifyTool` | POST JSON to any webhook URL |

### slack_message

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `channel` | string | yes | -- | Slack channel ID or name |
| `text` | string | yes | -- | Message text |
| `thread_ts` | string | no | None | Thread timestamp for replies |
| `token` | string | no | env `SLACK_BOT_TOKEN` | Slack bot token |

### slack_read

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `channel` | string | yes | -- | Slack channel ID |
| `count` | number | no | 10 | Number of messages to fetch |
| `token` | string | no | env `SLACK_BOT_TOKEN` | Slack bot token |

### discord_message

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `webhook_url` | string | yes | -- | Discord webhook URL |
| `content` | string | yes | -- | Message content |

### webhook_notify

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | yes | -- | Webhook URL |
| `payload` | object | yes | -- | JSON payload dict |
| `headers` | object | no | None | Custom HTTP headers |

---

## Browser Automation

Playwright-based browser automation tools. Requires `pip install playwright && playwright install chromium`. All tools share a singleton `BrowserSessionManager` that manages a headless Chromium browser.

| Tool | Class | Description |
|------|-------|-------------|
| `browser_navigate` | `BrowserNavigateTool` | Navigate to a URL |
| `browser_wait` | `BrowserWaitTool` | Wait for an element to reach a state |
| `browser_click` | `BrowserClickTool` | Click an element by CSS selector |
| `browser_type` | `BrowserTypeTool` | Type text into an input field |
| `browser_eval` | `BrowserEvalTool` | Execute JavaScript in page context |
| `browser_extract` | `BrowserExtractTool` | Extract page content as text or HTML |
| `browser_screenshot` | `BrowserScreenshotTool` | Take a page or element screenshot |

### browser_navigate

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | yes | -- | URL to navigate to |
| `wait_for` | string | no | `load` | Load state: `load`, `networkidle`, `domcontentloaded` |
| `timeout` | number | no | 30000 | Navigation timeout in milliseconds |

### browser_wait

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `selector` | string | yes | -- | CSS selector to wait for |
| `timeout` | number | no | 5000 | Max wait time in milliseconds |
| `state` | string | no | `visible` | Desired state: `visible`, `hidden`, `attached`, `detached` |

### browser_click

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `selector` | string | yes | -- | CSS selector of element to click |
| `timeout` | number | no | 5000 | Max wait time in milliseconds |

### browser_type

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `selector` | string | yes | -- | CSS selector of input field |
| `text` | string | yes | -- | Text to type |
| `clear_first` | boolean | no | true | Clear field before typing |

### browser_eval

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `script` | string | yes | -- | JavaScript code to evaluate |

### browser_extract

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `selector` | string | no | None | CSS selector (omit for whole page) |
| `format` | string | no | `text` | Output format: `text` or `html` |

### browser_screenshot

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `output_path` | string | yes | -- | File path to save PNG screenshot |
| `selector` | string | no | None | CSS selector of element (omit for full page) |
| `full_page` | boolean | no | false | Capture full scrollable page |

---

## Observability

Tools for tracing, logging, metrics recording, and LLM cost tracking.

| Tool | Class | Description |
|------|-------|-------------|
| `trace` | `TraceTool` | Create trace spans for distributed tracing |
| `performance_profile` | `PerformanceProfileTool` | Record tool execution timing and success |
| `log` | `LogTool` | Write structured JSON log entries |
| `metric` | `MetricTool` | Record custom metrics (counter, gauge, histogram) |
| `cost_tracker` | `CostTrackerTool` | Track LLM API call costs |

### trace

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | yes | -- | Trace span name |
| `attributes` | object | no | None | Key-value attributes |
| `parent_span_id` | string | no | None | Parent span ID for hierarchies |

### performance_profile

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tool_name` | string | yes | -- | Name of tool being profiled |
| `duration_ms` | number | yes | -- | Execution duration in milliseconds |
| `success` | boolean | yes | -- | Whether execution succeeded |
| `metadata` | object | no | None | Additional metadata |

Provides summary statistics via `PerformanceProfileTool.get_summary(tool_name)` with avg, min, max, p95, and error rate.

### log

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `level` | string | yes | -- | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `message` | string | yes | -- | Log message |
| `metadata` | object | no | None | Structured metadata |
| `log_path` | string | no | None | File path to append (JSON Lines format) |

### metric

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | yes | -- | Metric name |
| `value` | number | yes | -- | Metric value |
| `unit` | string | no | None | Unit (e.g. `ms`, `bytes`) |
| `tags` | object | no | None | Key-value tags |
| `metric_type` | string | no | `gauge` | Type: `counter` (accumulate), `gauge` (overwrite), `histogram` (append) |

### cost_tracker

Track LLM API costs with built-in pricing for common models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, etc.).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | yes | -- | Model name (e.g. `gpt-4o`, `claude-3-5-sonnet`) |
| `input_tokens` | number | yes | -- | Number of input tokens |
| `output_tokens` | number | yes | -- | Number of output tokens |
| `provider` | string | no | None | Provider name (e.g. `openai`, `anthropic`) |

```python
from quartermaster_tools.builtin.observability.cost import CostTrackerTool

tracker = CostTrackerTool()
tracker.run(model="gpt-4o", input_tokens=1000, output_tokens=500)
tracker.run(model="claude-3-5-sonnet", input_tokens=2000, output_tokens=1000)
print(CostTrackerTool.get_total_cost())       # cumulative USD cost
print(CostTrackerTool.get_cost_by_model())    # {"gpt-4o": ..., "claude-3-5-sonnet": ...}
```

---

## Privacy / PII Detection

Detect and redact personally identifiable information using regex patterns. Zero external dependencies.

| Tool | Class | Description |
|------|-------|-------------|
| `detect_pii` | `DetectPIITool` | Detect PII entities in text |
| `scan_file_pii` | `ScanFilePIITool` | Scan a file for PII entities |
| `redact_pii` | `RedactPIITool` | Redact PII from text with configurable strategy |

Supported PII entity types: `email`, `phone`, `credit_card` (with Luhn validation), `ssn`, `ip_address`, `date_of_birth`, `url_with_credentials`.

### detect_pii

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | yes | -- | Text to scan |
| `entities` | array | no | all types | Entity types to detect |
| `threshold` | number | no | 0.0 | Reserved for API compatibility |

```python
from quartermaster_tools.builtin.privacy.detect import DetectPIITool

tool = DetectPIITool()
result = tool.run(text="Contact john@example.com or call 555-123-4567")
# result.data["entities"] == [
#   {"type": "email", "value": "john@example.com", "start": 8, "end": 24},
#   {"type": "phone", "value": "555-123-4567", "start": 33, "end": 45},
# ]
```

### scan_file_pii

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | yes | -- | Path to file to scan |
| `entities` | array | no | all types | Entity types to detect |

Returns entities found plus `lines_with_pii` indicating which line numbers contain PII.

### redact_pii

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | yes | -- | Text to redact |
| `strategy` | string | no | `redact` | Strategy: `redact` (labels), `mask` (partial), `hash` (SHA-256) |
| `entities` | array | no | all types | Entity types to redact |

Strategies:
- `redact`: Replace with labels like `<EMAIL>`, `<PHONE>`, `<SSN>`
- `mask`: Partial masking, e.g. `j***@e******.com`, `***-**-4567`
- `hash`: Replace with first 8 characters of SHA-256 hash

```python
from quartermaster_tools.builtin.privacy.redact import RedactPIITool

tool = RedactPIITool()
result = tool.run(text="SSN: 123-45-6789", strategy="mask")
# result.data["redacted_text"] == "SSN: ***-**-6789"
```

---

## Compliance (EU AI Act)

Tools for EU AI Act risk classification, audit logging, and compliance checklists.

| Tool | Class | Description |
|------|-------|-------------|
| `risk_classifier` | `RiskClassifierTool` | Classify AI system risk level per EU AI Act |
| `audit_log` | `AuditLogTool` | Append to tamper-evident audit trail |
| `read_audit_log` | `ReadAuditLogTool` | Query and verify audit trail integrity |
| `compliance_checklist` | `ComplianceChecklistTool` | Generate compliance checklist by risk level |

### risk_classifier

Classifies an AI system as UNACCEPTABLE, HIGH, LIMITED, or MINIMAL risk per EU AI Act Annex III.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `system_description` | string | yes | -- | Description of the AI system |
| `domain` | string | yes | -- | Application domain (e.g. `healthcare`, `employment`, `education`, `law_enforcement`, `other`) |
| `uses_biometrics` | boolean | no | false | Whether biometric identification is used |
| `uses_subliminal_techniques` | boolean | no | false | Whether subliminal manipulation is used |
| `targets_vulnerable_groups` | boolean | no | false | Whether vulnerable groups are targeted |

Returns `risk_level`, `category` (Annex III reference), `obligations` (list of applicable articles), and `reasoning`.

### audit_log

Tamper-evident audit trail using JSON Lines. Each entry contains a SHA-256 hash of the previous entry forming a chain.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | yes | -- | Action being logged |
| `actor` | string | yes | -- | Who performed the action |
| `system_id` | string | yes | -- | AI system identifier |
| `details` | object | no | None | Additional details |
| `log_path` | string | no | `audit_log.jsonl` | Path to audit log file |

### read_audit_log

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `system_id` | string | yes | -- | Filter by system ID |
| `log_path` | string | no | `audit_log.jsonl` | Path to audit log file |
| `date_from` | string | no | None | Filter from ISO 8601 date |
| `date_to` | string | no | None | Filter to ISO 8601 date |
| `action_filter` | string | no | None | Filter by action type |
| `verify_integrity` | boolean | no | false | Verify hash chain integrity |

### compliance_checklist

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `risk_level` | string | yes | -- | Risk level: `UNACCEPTABLE`, `HIGH`, `LIMITED`, `MINIMAL` |
| `system_type` | string | no | None | Optional system type for context |

Returns a checklist of items, each with `article`, `requirement`, and `status` fields.

---

## A2A (Agent-to-Agent Protocol)

Tools for inter-agent communication using the A2A protocol (JSON-RPC 2.0 over HTTP). Enables discovering remote agents, sending tasks, and collecting results.

| Tool | Class | Description |
|------|-------|-------------|
| `a2a_discover` | `A2ADiscoverTool` | Fetch a remote agent's Agent Card |
| `a2a_register` | `A2ARegisterTool` | Generate an Agent Card for local registration |
| `a2a_send_task` | `A2ASendTaskTool` | Send a task to a remote A2A agent |
| `a2a_check_status` | `A2ACheckStatusTool` | Check the status of a remote A2A task |
| `a2a_collect_result` | `A2ACollectResultTool` | Collect results from a completed A2A task |

### a2a_discover

Fetch the Agent Card from a remote agent's `/.well-known/agent.json` endpoint. Returns the agent's name, description, skills, capabilities, and version. Includes SSRF protection against private/reserved IP networks.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agent_url` | string | yes | -- | Base URL of the remote A2A agent |

Constructor accepts `timeout` (default 30 seconds).

Returns: `{"agent_name", "description", "skills", "capabilities", "version"}`.

Requires httpx (`pip install quartermaster-tools[web]`).

```python
from quartermaster_tools.builtin.a2a.discover import A2ADiscoverTool

tool = A2ADiscoverTool(timeout=30)
result = tool.run(agent_url="https://agent.example.com")
# result.data == {
#     "agent_name": "SummaryBot",
#     "description": "Summarizes documents",
#     "skills": [{"id": "summarize", "name": "Summarize", ...}],
#     "capabilities": {"streaming": False, "pushNotifications": False},
#     "version": "1.0.0",
# }
```

### a2a_register

Generate an A2A Agent Card JSON document describing the local agent. Optionally writes the card to a file. No external dependencies required.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | yes | -- | Name of the agent |
| `description` | string | yes | -- | Short description of the agent |
| `url` | string | yes | -- | Public URL where the agent is reachable |
| `skills` | array | yes | -- | List of skill dicts with `id`, `name`, `description` keys |
| `version` | string | no | `"1.0.0"` | Agent version string |
| `streaming` | boolean | no | `false` | Whether the agent supports streaming responses |
| `push_notifications` | boolean | no | `false` | Whether the agent supports push notifications |
| `output_path` | string | no | None | Optional file path to save the Agent Card JSON |

Returns: `{"agent_card": {...}, "saved_to": "path" or null}`.

```python
from quartermaster_tools.builtin.a2a.register import A2ARegisterTool

tool = A2ARegisterTool()
result = tool.run(
    name="AnalysisAgent",
    description="Performs data analysis",
    url="https://analysis.example.com",
    skills=[{"id": "analyze", "name": "Analyze", "description": "Run analysis on data"}],
    output_path=".well-known/agent.json",
)
# result.data["saved_to"] == ".well-known/agent.json"
```

### a2a_send_task

Send a task to a remote A2A agent via JSON-RPC 2.0 `tasks/send`. Includes SSRF protection. Auto-generates a UUID task ID if not provided.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agent_url` | string | yes | -- | Base URL of the remote A2A agent |
| `task_message` | string | yes | -- | The task instruction to send |
| `task_id` | string | no | auto uuid4 | Optional task ID |
| `metadata` | object | no | `{}` | Optional metadata dict to include with the task |

Constructor accepts `timeout` (default 60 seconds).

Returns: `{"task_id", "status", "artifacts"}`.

Requires httpx (`pip install quartermaster-tools[web]`).

### a2a_check_status

Check the status of a previously sent A2A task via JSON-RPC 2.0 `tasks/get`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agent_url` | string | yes | -- | Base URL of the remote A2A agent |
| `task_id` | string | yes | -- | ID of the task to check |

Returns: `{"task_id", "status", "artifacts"}`.

### a2a_collect_result

Collect and format results from a completed A2A task. Extracts text content from artifact parts.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agent_url` | string | yes | -- | Base URL of the remote A2A agent |
| `task_id` | string | yes | -- | ID of the task whose results to collect |

Returns: `{"task_id", "completed": bool, "results": [{"type", "content"}], "status"}`.

```python
from quartermaster_tools.builtin.a2a.task import A2ASendTaskTool, A2ACollectResultTool

send = A2ASendTaskTool(timeout=60)
result = send.run(agent_url="https://agent.example.com", task_message="Summarize this report")
task_id = result.data["task_id"]

collect = A2ACollectResultTool(timeout=60)
output = collect.run(agent_url="https://agent.example.com", task_id=task_id)
# output.data["results"] == [{"type": "text", "content": "Summary: ..."}]
```

---

## Agent Sessions

Tools for managing parallel agent sessions with threading support. Sessions track status, messages, results, and support finish hooks. All tools share a `SessionManager` (default singleton or custom instance).

Session statuses: `created`, `running`, `waiting`, `completed`, `failed`, `cancelled`.

| Tool | Class | Description |
|------|-------|-------------|
| `spawn_agent` | `SpawnAgentTool` | Create and start an agent session in one step |
| `create_agent_session` | `CreateSessionTool` | Create a session (without starting) |
| `start_agent_session` | `StartSessionTool` | Start a previously created session with a task |
| `inject_agent_message` | `InjectMessageTool` | Inject a message into a running session |
| `get_agent_session_status` | `GetSessionStatusTool` | Get detailed status of a session |
| `list_agent_sessions` | `ListSessionsTool` | List all sessions, optionally filtered by status |
| `wait_agent_session` | `WaitSessionTool` | Block until a session completes |
| `collect_agent_results` | `CollectResultsTool` | Wait for and collect results from multiple sessions |
| `cancel_agent_session` | `CancelSessionTool` | Mark a session as cancelled |
| `add_agent_finish_hook` | `AddFinishHookTool` | Register a callback for session completion |
| `notify_parent` | `NotifyParentTool` | Send status updates from a sub-agent to its parent |

### spawn_agent

The preferred tool for simple agent spawning. Combines session creation and start into a single call. Supports an `allowed_agents` whitelist for security.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agent_id` | string | yes | -- | Which agent to spawn (validated against allowed list) |
| `task` | string | yes | -- | Task description/instructions for the agent |
| `name` | string | no | `""` | Optional human-readable session name |
| `system_prompt` | string | no | `""` | Optional system prompt override |
| `allowed_agents` | string | no | `""` | Comma-separated agent IDs this agent can itself spawn |
| `parent_session_id` | string | no | `""` | Parent session ID for notification routing |

Constructor accepts optional `manager` (SessionManager) and `allowed_agents` (list of allowed agent IDs; empty means allow all).

Returns: `{"session_id", "status": "running"}`.

```python
from quartermaster_tools.builtin.agents.tools import SpawnAgentTool

tool = SpawnAgentTool(allowed_agents=["researcher", "writer"])
result = tool.run(agent_id="researcher", task="Find recent papers on RAG")
session_id = result.data["session_id"]
```

### create_agent_session / start_agent_session

For advanced patterns where you need to pre-configure a session (inject messages, add hooks) before starting it.

**create_agent_session:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | no | `""` | Human-readable name |
| `metadata` | string | no | `""` | JSON string of metadata key-value pairs |

Returns: `{"session_id", "name", "status"}`.

**start_agent_session:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | yes | -- | The session ID to start |
| `task` | string | yes | -- | Task description/instructions |
| `system_prompt` | string | no | `""` | Optional system prompt |

Returns: `{"session_id", "status": "running"}`.

### inject_agent_message

Add a message to a running session's history. Useful for providing additional context mid-execution.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | yes | -- | The session ID to inject into |
| `content` | string | yes | -- | Message content |
| `role` | string | no | `"user"` | Message role: `user`, `assistant`, or `system` |

Returns: `{"session_id", "injected": true, "message_count"}`.

### get_agent_session_status

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | yes | -- | The session ID to check |

Returns: `{"session_id", "status", "name", "message_count", "created_at", "updated_at"}`.

### list_agent_sessions

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `status` | string | no | `""` | Filter by status: `created`, `running`, `completed`, `failed`, `cancelled` |

Returns: `{"sessions": [{id, name, status, message_count, created_at}], "count"}`.

### wait_agent_session

Blocks until the session finishes or the timeout expires.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | yes | -- | The session ID to wait for |
| `timeout` | number | no | 30 | Maximum seconds to wait |

Returns: `{"session_id", "status", "result", "error"}`.

### collect_agent_results

Wait for multiple sessions and collect all results.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_ids` | string | yes | -- | Comma-separated list of session IDs |
| `timeout` | number | no | 30 | Maximum seconds to wait per session |

Returns: `{"results": [{session_id, status, result, error}], "all_completed": bool}`.

### cancel_agent_session

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | yes | -- | The session ID to cancel |

Returns: `{"session_id", "cancelled": true}`.

Note: Sets the status flag to cancelled. The underlying thread may still be running.

### add_agent_finish_hook

Register a callback that fires when a session completes.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | yes | -- | The session ID to add a hook to |
| `hook_type` | string | yes | -- | Type of hook: `log` or `notify` |
| `hook_config` | string | no | `""` | Optional JSON string of hook configuration |

Built-in hook types:
- `log` -- Appends a line to a log file (config: `{"path": "agent_session.log"}`)
- `notify` -- Stores a notification dict in session metadata

Returns: `{"session_id", "hook_added": true, "hook_type"}`.

### notify_parent

Sub-agents running in background sessions call this to send status updates back to the parent agent.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `message` | string | yes | -- | Message to send to parent agent |
| `status` | string | no | `"progress"` | Status: `progress`, `completed`, `failed` |
| `data` | string | no | `""` | JSON string of additional data |

Returns: `{"notification": {message, status, data, timestamp}, "status"}`.

```python
from quartermaster_tools.builtin.agents.tools import (
    SpawnAgentTool, WaitSessionTool, CollectResultsTool,
)

spawn = SpawnAgentTool(allowed_agents=["researcher", "writer"])

# Spawn two agents in parallel
r1 = spawn.run(agent_id="researcher", task="Find papers on RAG")
r2 = spawn.run(agent_id="writer", task="Draft an outline")

# Collect all results
collect = CollectResultsTool()
results = collect.run(
    session_ids=f"{r1.data['session_id']},{r2.data['session_id']}",
    timeout=60,
)
# results.data["all_completed"] == True
```

---

## Creating Custom Tools with @tool

The `@tool` decorator converts plain Python functions into `FunctionTool` instances. It extracts metadata from the function signature, type hints, and Google-style docstrings.

```python
from quartermaster_tools import tool

@tool()
def summarize_text(text: str, max_length: int = 100) -> dict:
    """Summarize a block of text.

    Takes input text and returns a shortened version.

    Args:
        text: The text to summarize.
        max_length: Maximum length of the summary.

    Returns:
        A dict with the summary.
    """
    summary = text[:max_length].rsplit(" ", 1)[0] + "..."
    return {"summary": summary}

# summarize_text is now a FunctionTool instance
print(summarize_text.name())        # "summarize_text"
print(summarize_text.parameters())  # [ToolParameter(name="text", ...), ToolParameter(name="max_length", ...)]

# Call it directly
result = summarize_text("This is a long piece of text...")

# Or via the tool interface
result = summarize_text.run(text="This is a long piece of text...", max_length=20)
# result.success == True, result.data == {"summary": "This is a long..."}
```

You can override the name and description:

```python
@tool(name="custom_summarizer", description="Custom text summarization tool")
def my_func(text: str) -> dict:
    ...
```

The decorator supports:
- Sync and async functions
- All Python type hints mapped to tool parameter types (`str` -> `string`, `int` -> `integer`, `float` -> `number`, `bool` -> `boolean`, `list` -> `array`, `dict` -> `object`)
- Default values for optional parameters
- Context parameters (`ctx`, `context`, `self`, `cls`) are automatically skipped

---

## Registering Custom Tools

### Using the ToolRegistry

```python
from quartermaster_tools import ToolRegistry, PythonExecutorTool, EvalMathTool

registry = ToolRegistry()

# Register tool instances
registry.register(PythonExecutorTool())
registry.register(EvalMathTool())

# Look up tools
tool = registry.get("python_executor")
tool = registry.get("eval_math", version="1.0.0")

# List all tools
for descriptor in registry.list_tools():
    print(f"{descriptor.name}: {descriptor.short_description}")
```

### Using the registry decorator

```python
registry = ToolRegistry()

@registry.tool()
def fetch_weather(city: str) -> dict:
    """Fetch current weather for a city."""
    return {"city": city, "temp": 72}

# Tool is now registered and accessible
tool = registry.get("fetch_weather")
```

### Using the class decorator with the default registry

```python
from quartermaster_tools import register_tool, get_default_registry, AbstractTool

@register_tool
class MyCustomTool(AbstractTool):
    def name(self): return "my_custom"
    def version(self): return "1.0.0"
    def parameters(self): return []
    def info(self): ...
    def run(self, **kwargs): ...

# Accessible from the default registry
registry = get_default_registry()
tool = registry.get("my_custom")
```

### Plugin discovery via entry points

Tools can be auto-discovered from installed packages using Python entry points:

```toml
# In your package's pyproject.toml
[project.entry-points.quartermaster_tools]
my_tool = "my_package.tools:MyTool"
```

The registry automatically loads plugins on first access via `importlib.metadata.entry_points()`.

---

## Bridging Tools to LLM Providers

The `ToolRegistry` can export all registered tools in the format required by major LLM providers for function calling / tool use.

### OpenAI format

```python
registry = ToolRegistry()
registry.register(EvalMathTool())
registry.register(DuckDuckGoSearchTool())

openai_tools = registry.to_openai_tools()
# [
#   {
#     "type": "function",
#     "function": {
#       "name": "eval_math",
#       "description": "Safely evaluate mathematical expressions.",
#       "parameters": {
#         "type": "object",
#         "properties": {"expression": {"type": "string", "description": "..."}},
#         "required": ["expression"]
#       }
#     }
#   },
#   ...
# ]
```

### Anthropic format

```python
anthropic_tools = registry.to_anthropic_tools()
# [
#   {
#     "name": "eval_math",
#     "description": "Safely evaluate mathematical expressions.",
#     "input_schema": {
#       "type": "object",
#       "properties": {"expression": {"type": "string", "description": "..."}},
#       "required": ["expression"]
#     }
#   },
#   ...
# ]
```

### MCP (Model Context Protocol) format

```python
mcp_tools = registry.to_mcp_tools()
# [
#   {
#     "name": "eval_math",
#     "description": "Safely evaluate mathematical expressions.",
#     "inputSchema": {
#       "type": "object",
#       "properties": {"expression": {"type": "string", "description": "..."}},
#       "required": ["expression"]
#     }
#   },
#   ...
# ]
```

### Raw JSON Schema

```python
schemas = registry.to_json_schema()
# Returns list of {"name": ..., "description": ..., "parameters": {...}} dicts
```

All export methods automatically convert `ToolParameter` type strings to JSON Schema types and include `required` arrays, `default` values, and `enum` options where applicable.

## See Also

- [Tool System](tools.md) -- Tool architecture, AbstractTool, ToolRegistry, custom tool development
- [Graph Building](graph-building.md) -- Using tool nodes in agent graphs
- [Security](security.md) -- Tool parameter validation and safe execution
- [Engine](engine.md) -- How the engine executes tool nodes at runtime
