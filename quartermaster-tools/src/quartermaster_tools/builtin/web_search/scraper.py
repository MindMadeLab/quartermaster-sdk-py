"""
Web scraper: fetch and convert web pages to readable text.

Supports output in plain text, basic markdown, or raw HTML. Uses regex-based
HTML stripping to avoid external dependencies like BeautifulSoup.
"""

from __future__ import annotations

import re

from quartermaster_tools.builtin.web_request import _validate_url
from quartermaster_tools.decorator import tool

_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120
_MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10 MB

# Regex patterns for HTML processing
_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\n{3,}")
_SPACES_RE = re.compile(r"[ \t]+")

# Markdown conversion patterns
_HEADING_RE = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.DOTALL | re.IGNORECASE)
_LINK_RE = re.compile(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
_BOLD_RE = re.compile(r"<(b|strong)(?:\s[^>]*)?>(.+?)</\1>", re.DOTALL | re.IGNORECASE)
_ITALIC_RE = re.compile(r"<(i|em)(?:\s[^>]*)?>(.+?)</\1>", re.DOTALL | re.IGNORECASE)
_LI_RE = re.compile(r"<li[^>]*>(.*?)</li>", re.DOTALL | re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL | re.IGNORECASE)
_IMG_RE = re.compile(r'<img[^>]*alt="([^"]*)"[^>]*/?>', re.IGNORECASE)


def _strip_to_text(html_content: str) -> str:
    """Strip HTML to plain readable text.

    Args:
        html_content: Raw HTML string.

    Returns:
        Plain text with tags removed and whitespace normalized.
    """
    import html as html_mod

    text = _SCRIPT_STYLE_RE.sub("", html_content)
    text = _COMMENT_RE.sub("", text)
    # Convert block elements to newlines
    text = re.sub(r"<(br|p|div|h[1-6]|li|tr|blockquote)[^>]*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</(p|div|h[1-6]|li|tr|blockquote|ul|ol|table)>", "\n", text, flags=re.IGNORECASE
    )
    text = _TAG_RE.sub("", text)
    text = html_mod.unescape(text)
    text = _SPACES_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub("\n\n", text)
    return text.strip()


def _strip_to_markdown(html_content: str) -> str:
    """Convert HTML to basic markdown.

    Args:
        html_content: Raw HTML string.

    Returns:
        Markdown-formatted text.
    """
    import html as html_mod

    text = _SCRIPT_STYLE_RE.sub("", html_content)
    text = _COMMENT_RE.sub("", text)

    # Convert headings
    def _heading_repl(m: re.Match) -> str:
        level = int(m.group(1))
        content = _TAG_RE.sub("", m.group(2)).strip()
        return f"\n{'#' * level} {content}\n"

    # Convert inline elements first (bold, italic) before block elements
    text = _BOLD_RE.sub(r"**\2**", text)
    text = _ITALIC_RE.sub(r"*\2*", text)

    text = _HEADING_RE.sub(_heading_repl, text)

    # Convert links
    def _link_repl(m: re.Match) -> str:
        href = m.group(1)
        label = _TAG_RE.sub("", m.group(2)).strip()
        return f"[{label}]({href})"

    text = _LINK_RE.sub(_link_repl, text)

    # Convert images
    text = _IMG_RE.sub(r"![\1]", text)

    # Convert list items
    text = _LI_RE.sub(r"\n- \1", text)

    # Convert <br> and <p>
    text = _BR_RE.sub("\n", text)
    text = _P_RE.sub(r"\n\1\n", text)

    # Strip remaining tags
    text = re.sub(r"</(p|div|ul|ol|table|tr)>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub("", text)
    text = html_mod.unescape(text)
    text = _SPACES_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub("\n\n", text)
    return text.strip()


@tool()
def web_scraper(url: str, output_format: str = "text", timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """Fetch a web page and return its content as text, markdown, or HTML.

    Fetches a URL using httpx and converts the HTML content to the
    requested output format. Supports plain text (tags stripped),
    basic markdown conversion, or raw HTML. Uses regex-based parsing
    with no external HTML library dependencies.

    Args:
        url: The URL to scrape.
        output_format: Output format: text, markdown, or html.
        timeout: Request timeout in seconds (default 30, max 120).
    """
    url = url.strip() if url else ""
    output_format = output_format.lower() if output_format else "text"
    timeout = min(int(timeout), _MAX_TIMEOUT)

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
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; QuartermasterBot/1.0)",
                },
            )
            response.raise_for_status()

            if len(response.content) > _MAX_RESPONSE_SIZE:
                raise ValueError(
                    f"Response too large: {len(response.content)} bytes (limit: {_MAX_RESPONSE_SIZE})"
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
        content = _strip_to_markdown(raw_html)
    else:
        content = _strip_to_text(raw_html)

    return {
        "content": content,
        "url": url,
        "content_length": len(content),
    }
