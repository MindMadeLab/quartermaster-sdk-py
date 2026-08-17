"""
Web scraper: fetch and convert web pages to readable text.

Supports output in plain text, markdown, or raw HTML. HTML-to-text/markdown
conversion uses the stdlib ``html.parser`` (no extra required dependencies).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from quartermaster_tools.builtin.web_request import _validate_url
from quartermaster_tools.decorator import tool

_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120
_MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10 MB
_DEFAULT_MAX_CHARS = 100_000
_MAX_REDIRECTS = 10
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
_TRUNCATION_MARKER = "\n\n[truncated]"

# Realistic browser UA: bot-like strings often get challenge pages.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})
_VOID_TAGS = frozenset(
    {
        "br",
        "hr",
        "img",
        "input",
        "meta",
        "link",
        "area",
        "base",
        "col",
        "embed",
        "source",
        "track",
        "wbr",
    }
)
_HEADING_TAGS = frozenset({f"h{i}" for i in range(1, 7)})
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_SPACE_RE = re.compile(r" {2,}")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _attr_map(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
    """Lower-case HTML attributes, dropping None values."""
    return {name.lower(): value for name, value in attrs if value is not None}


def _canonical_host(hostname: str | None) -> str:
    """Lower-case host with a single leading ``www.`` stripped."""
    if not hostname:
        return ""
    host = hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_same_host(original_url: str, target_url: str) -> bool:
    """True if both URLs share a host, treating www add/remove as equivalent."""
    return _canonical_host(urlparse(original_url).hostname) == _canonical_host(
        urlparse(target_url).hostname
    )


class _HtmlConverter(HTMLParser):
    """Convert HTML to plain text or GitHub-flavored-ish markdown."""

    def __init__(self, *, markdown: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.markdown = markdown
        self._buffers: list[list[str]] = [[]]
        self._skip_depth = 0
        self._pre_depth = 0
        self._code_depth = 0
        self._list_kinds: list[str] = []
        self._ol_counters: list[int] = []
        self._at_li_start = False
        self._table_rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._link_hrefs: list[str | None] = []
        self._heading_levels: list[int] = []
        self._code_langs: list[str] = []
        self._blockquote_depth = 0

    def convert(self, html_content: str) -> str:
        self.feed(html_content)
        self.close()
        return _finalize_output("".join(self._buffers[0]))

    def _emit(self, text: str) -> None:
        if text:
            self._buffers[-1].append(text)

    def _push(self) -> None:
        self._buffers.append([])

    def _pop(self) -> str:
        # Unmatched end tags must not steal the document buffer.
        if len(self._buffers) == 1:
            return ""
        return "".join(self._buffers.pop())

    def _last_char(self) -> str:
        buf = self._buffers[-1]
        for chunk in reversed(buf):
            if chunk:
                return chunk[-1]
        return ""

    def _ensure_newlines(self, count: int = 1) -> None:
        if self._pre_depth:
            return
        needed = count
        buf = self._buffers[-1]
        trailing = 0
        for chunk in reversed(buf):
            for ch in reversed(chunk):
                if ch == "\n":
                    trailing += 1
                else:
                    needed = max(0, count - trailing)
                    if needed:
                        self._emit("\n" * needed)
                    return
            # chunk was all newlines; keep counting
        needed = max(0, count - trailing)
        if needed:
            self._emit("\n" * needed)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        attrs_d = _attr_map(attrs)

        if tag == "br":
            self._emit("\n")
            if self._blockquote_depth and self.markdown:
                self._emit("> ")
            return
        if tag == "hr":
            self._ensure_newlines(2)
            self._emit("---" if self.markdown else "")
            self._ensure_newlines(2)
            return
        if tag == "img":
            alt = attrs_d.get("alt", "")
            src = attrs_d.get("src", "")
            if self.markdown:
                self._emit(f"![{alt}]({src})" if src else f"![{alt}]")
            elif alt:
                self._emit(alt)
            return

        if tag in _HEADING_TAGS:
            self._heading_levels.append(int(tag[1]))
            self._push()
            return
        if tag == "a":
            self._link_hrefs.append(attrs_d.get("href"))
            self._push()
            return
        if tag in {"b", "strong"}:
            self._push()
            return
        if tag in {"i", "em"}:
            self._push()
            return
        if tag == "code":
            lang = _language_from_class(attrs_d.get("class", ""))
            if self._pre_depth:
                if lang and self._code_langs:
                    self._code_langs[-1] = lang
                return
            self._code_langs.append(lang)
            self._code_depth += 1
            self._push()
            return
        if tag == "pre":
            self._pre_depth += 1
            lang = _language_from_class(attrs_d.get("class", ""))
            self._code_langs.append(lang)
            self._push()
            return
        if tag in {"ul", "ol"}:
            self._list_kinds.append(tag)
            self._ol_counters.append(0)
            return
        if tag == "li":
            indent = "  " * max(0, len(self._list_kinds) - 1)
            kind = self._list_kinds[-1] if self._list_kinds else "ul"
            if kind == "ol":
                if self._ol_counters:
                    self._ol_counters[-1] += 1
                    n = self._ol_counters[-1]
                else:
                    n = 1
                marker = f"{n}. "
            else:
                marker = "- "
            self._ensure_newlines(1)
            self._emit(f"{indent}{marker}")
            self._at_li_start = True
            return
        if tag == "tr":
            self._current_row = []
            return
        if tag in {"td", "th"}:
            self._push()
            return
        if tag == "blockquote":
            self._blockquote_depth += 1
            self._ensure_newlines(2)
            if self.markdown:
                self._emit("> ")
            return
        if tag == "p":
            if self._at_li_start:
                self._at_li_start = False
            else:
                self._ensure_newlines(2)
            return
        if tag in {"div", "section", "article", "header", "footer", "main", "nav", "figure"}:
            self._ensure_newlines(1)
            return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in _VOID_TAGS:
            return

        if tag in _HEADING_TAGS:
            level = self._heading_levels.pop() if self._heading_levels else int(tag[1])
            text = _collapse_inline(self._pop()).strip()
            self._ensure_newlines(2)
            if self.markdown and text:
                self._emit(f"{'#' * level} {text}")
            else:
                self._emit(text)
            self._ensure_newlines(2)
            return
        if tag == "a":
            href = self._link_hrefs.pop() if self._link_hrefs else None
            text = _collapse_inline(self._pop()).strip()
            if self.markdown and href:
                self._emit(f"[{text}]({href})" if text else f"<{href}>")
            elif text and href and not self.markdown:
                self._emit(f"{text} ({href})")
            else:
                self._emit(text)
            return
        if tag in {"b", "strong"}:
            text = self._pop()
            if self.markdown and text.strip():
                self._emit(f"**{text.strip()}**")
            else:
                self._emit(text)
            return
        if tag in {"i", "em"}:
            text = self._pop()
            if self.markdown and text.strip():
                self._emit(f"*{text.strip()}*")
            else:
                self._emit(text)
            return
        if tag == "code" and self._pre_depth == 0 and self._code_depth:
            self._code_depth -= 1
            if self._code_langs:
                self._code_langs.pop()
            text = self._pop()
            if self.markdown and text:
                fence = "``" if "`" in text else "`"
                self._emit(f"{fence}{text}{fence}")
            else:
                self._emit(text)
            return
        if tag == "pre":
            if self._pre_depth:
                self._pre_depth -= 1
            lang = self._code_langs.pop() if self._code_langs else ""
            text = self._pop().strip("\n")
            self._ensure_newlines(2)
            if self.markdown:
                self._emit(f"```{lang}\n{text}\n```")
            else:
                self._emit(text)
            self._ensure_newlines(2)
            return
        if tag in {"ul", "ol"}:
            if self._list_kinds:
                self._list_kinds.pop()
            if self._ol_counters:
                self._ol_counters.pop()
            self._ensure_newlines(2)
            return
        if tag == "li":
            self._at_li_start = False
            self._ensure_newlines(1)
            return
        if tag in {"td", "th"}:
            cell = _collapse_inline(self._pop()).strip().replace("\n", " ")
            if self._current_row is not None:
                self._current_row.append(cell)
            else:
                self._emit(cell)
            return
        if tag == "tr":
            if self._current_row is not None:
                self._table_rows.append(self._current_row)
                self._current_row = None
            return
        if tag == "table":
            self._flush_table()
            return
        if tag == "blockquote":
            if self._blockquote_depth:
                self._blockquote_depth -= 1
            self._ensure_newlines(2)
            return
        if tag in {"p", "div", "section", "article", "header", "footer", "main", "nav", "figure"}:
            self._ensure_newlines(2)
            return

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data:
            return
        self._at_li_start = False
        if self._pre_depth:
            self._emit(data)
            return
        collapsed = _WS_RE.sub(" ", data.replace("\n", " "))
        if collapsed == " " and self._last_char() in {"", " ", "\n"}:
            return
        if collapsed.startswith(" ") and self._last_char() in {"", " ", "\n"}:
            collapsed = collapsed.lstrip(" ")
            if not collapsed:
                return
        self._emit(collapsed)

    def handle_comment(self, data: str) -> None:
        return

    def _flush_table(self) -> None:
        rows = [row for row in self._table_rows if any(c.strip() for c in row)]
        self._table_rows = []
        self._current_row = None
        if not rows:
            return
        width = max(len(row) for row in rows)
        padded = [row + [""] * (width - len(row)) for row in rows]

        def _esc(cell: str) -> str:
            return cell.replace("|", "\\|")

        self._ensure_newlines(2)
        for i, row in enumerate(padded):
            cells = [_esc(c) for c in row]
            self._emit("| " + " | ".join(cells) + " |")
            self._emit("\n")
            if i == 0:
                self._emit("| " + " | ".join("---" for _ in cells) + " |")
                self._emit("\n")
        self._ensure_newlines(2)


def _language_from_class(class_attr: str) -> str:
    """Extract a fenced-code language from a class attribute, if present."""
    for part in class_attr.split():
        if part.startswith("language-"):
            return part[len("language-") :]
        if part.startswith("lang-"):
            return part[len("lang-") :]
    return ""


def _collapse_inline(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", text.replace("\t", " ")).strip()


def _finalize_output(text: str) -> str:
    """Normalize whitespace while preserving list indent and fenced code."""
    lines: list[str] = []
    in_fence = False
    blank_run = 0
    for raw in text.splitlines():
        if raw.strip().startswith("```"):
            in_fence = not in_fence
            blank_run = 0
            lines.append(raw.rstrip())
            continue
        if in_fence:
            lines.append(raw.rstrip())
            blank_run = 0
            continue
        stripped_end = raw.rstrip()
        if not stripped_end.strip():
            blank_run += 1
            if blank_run <= 1:
                lines.append("")
            continue
        blank_run = 0
        lead = len(stripped_end) - len(stripped_end.lstrip(" "))
        indent = stripped_end[:lead]
        rest = _MULTI_SPACE_RE.sub(" ", stripped_end[lead:])
        lines.append(indent + rest)
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def _html_to_text(html_content: str) -> str:
    """Strip HTML to plain readable text."""
    return _HtmlConverter(markdown=False).convert(html_content)


def _html_to_markdown(html_content: str) -> str:
    """Convert HTML to markdown using stdlib html.parser."""
    return _HtmlConverter(markdown=True).convert(html_content)


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending a marker when clipped."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = _TRUNCATION_MARKER
    if max_chars <= len(marker):
        return marker[:max_chars]
    return text[: max_chars - len(marker)] + marker


def _header_get(headers: object, name: str) -> str | None:
    """Read a header from a dict or httpx-like case-insensitive map."""
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    value = getter(name)
    if value is None and name.lower() != name:
        value = getter(name.lower())
    if value is None:
        value = getter(name.title())
    return value if value is None else str(value)


def _fetch_with_same_host_redirects(client: object, url: str) -> tuple[object, str]:
    """GET ``url``, following only same-host redirects (www add/remove allowed).

    Returns:
        (response, final_url)

    Raises:
        ValueError: SSRF / URL validation failure (including redirect targets).
        RuntimeError: Cross-host redirect, missing Location, or hop cap exceeded.
    """
    current = url
    original = url
    get = client.get  # type: ignore[attr-defined]
    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }

    for _hop in range(_MAX_REDIRECTS + 1):
        url_error = _validate_url(current)
        if url_error:
            raise ValueError(url_error)

        response = get(current, headers=headers)
        status = int(getattr(response, "status_code", 0))
        if status in _REDIRECT_STATUS:
            location = _header_get(getattr(response, "headers", {}), "location")
            if not location:
                raise RuntimeError(
                    f"HTTP {status} redirect from {original} missing Location header"
                )
            next_url = urljoin(current, location.strip())
            if not _is_same_host(original, next_url):
                raise RuntimeError(
                    "Redirect to another host was detected: "
                    f"original URL {original} redirected to {next_url} "
                    f"(HTTP {status})"
                )
            next_error = _validate_url(next_url)
            if next_error:
                raise ValueError(next_error)
            current = next_url
            continue

        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()
        return response, current

    raise RuntimeError(f"Too many redirects (max {_MAX_REDIRECTS}) starting from {original}")


@tool()
def web_scraper(
    url: str,
    output_format: str = "text",
    timeout: int = _DEFAULT_TIMEOUT,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> dict:
    """Fetch a web page and return its content as text, markdown, or HTML.

    Fetches a URL using httpx and converts HTML to the requested format.
    Markdown/text conversion uses the stdlib html.parser (tables, nested
    lists, headings, links, emphasis, and code blocks). Same-host redirects
    are followed (www add/remove allowed); cross-host redirects are rejected.
    Converted text/markdown is truncated at max_chars.

    Args:
        url: The URL to scrape.
        output_format: Output format: text, markdown, or html.
        timeout: Request timeout in seconds (default 30, max 120).
        max_chars: Max characters of converted text/markdown (default 100000).
            HTML is still capped by the 10 MB response-size limit.
    """
    url = url.strip() if url else ""
    output_format = output_format.lower() if output_format else "text"
    timeout = min(int(timeout), _MAX_TIMEOUT)
    max_chars = int(max_chars)

    if not url:
        raise ValueError("Parameter 'url' is required")

    # SSRF protection: block private/internal network access
    url_error = _validate_url(url)
    if url_error:
        raise ValueError(url_error)

    if output_format not in ("text", "markdown", "html"):
        raise ValueError(
            f"Invalid output_format: {output_format!r}. Use 'text', 'markdown', or 'html'."
        )

    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx is required for WebScraperTool. "
            "Install it with: pip install quartermaster-tools[web]"
        )

    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response, final_url = _fetch_with_same_host_redirects(client, url)

            content_bytes = getattr(response, "content", b"")
            if len(content_bytes) > _MAX_RESPONSE_SIZE:
                raise ValueError(
                    f"Response too large: {len(content_bytes)} bytes (limit: {_MAX_RESPONSE_SIZE})"
                )

            raw_html = response.text
    except httpx.TimeoutException:
        raise TimeoutError(f"Request timed out after {timeout} seconds")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP {e.response.status_code}: {e}")
    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error: {e}")

    if output_format == "html":
        content = raw_html
    elif output_format == "markdown":
        content = _truncate(_html_to_markdown(raw_html), max_chars)
    else:
        content = _truncate(_html_to_text(raw_html), max_chars)

    return {
        "content": content,
        "url": final_url,
        "content_length": len(content),
    }
