"""
PII detection tools: detect_pii_tool and scan_file_pii.

Detect personally identifiable information in text or files using
regex patterns. No external dependencies required.
"""

from __future__ import annotations

import os
import re
from typing import Any

from quartermaster_tools.decorator import tool

# Blocked path prefixes for file scanning
_BLOCKED_PREFIXES = (
    "/etc/",
    "/private/etc/",
    "/proc/",
    "/sys/",
    "/dev/",
    "/boot/",
    "/var/run/secrets/",
)

# PII detection patterns
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    ),
    "phone": re.compile(
        r"(?<!\d)"
        r"(?:"
        r"\+?1[\s\-.]?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"  # US: +1 (555) 123-4567
        r"|"
        r"\+[1-9]\d{0,2}[\s\-.]?\(?\d{1,4}\)?[\s\-.]?\d{2,4}[\s\-.]?\d{2,4}"  # International: +XX ...
        r")"
        r"(?!\d)",
    ),
    "credit_card": re.compile(
        r"(?<!\d)"
        r"(?:"
        r"4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}"  # Visa
        r"|"
        r"5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}"  # Mastercard
        r"|"
        r"3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5}"  # Amex
        r"|"
        r"6(?:011|5\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}"  # Discover
        r")"
        r"(?!\d)",
    ),
    "ssn": re.compile(
        r"(?<!\d)"
        r"(?!000|666|9\d{2})\d{3}"
        r"[\s\-]"
        r"(?!00)\d{2}"
        r"[\s\-]"
        r"(?!0000)\d{4}"
        r"(?!\d)",
    ),
    "ip_address": re.compile(
        r"(?<!\d)"
        r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
        r"(?!\d)",
    ),
    "date_of_birth": re.compile(
        r"(?i)"
        r"(?:"
        r"(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}"  # MM/DD/YYYY
        r"|"
        r"(?:0[1-9]|[12]\d|3[01])[/\-](?:0[1-9]|1[0-2])[/\-](?:19|20)\d{2}"  # DD/MM/YYYY
        r"|"
        r"(?:19|20)\d{2}[/\-](?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])"  # YYYY/MM/DD
        r")",
    ),
    "url_with_credentials": re.compile(
        r"https?://[a-zA-Z0-9._%+\-]+:[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}[^\s]*",
    ),
}


def _luhn_check(number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse = digits[::-1]
    for i, d in enumerate(reverse):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _overlaps(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
    """Check if a range overlaps any occupied range."""
    for os_, oe in occupied:
        if start < oe and end > os_:
            return True
    return False


# Detection priority: more specific patterns first to avoid phone matching SSN/CC
_DETECTION_ORDER = [
    "url_with_credentials",
    "email",
    "credit_card",
    "ssn",
    "date_of_birth",
    "ip_address",
    "phone",
]


def detect_pii(
    text: str,
    entities: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Detect PII entities in text.

    Args:
        text: The text to scan.
        entities: Optional list of entity types to filter.

    Returns:
        List of detected entity dicts with type, value, start, end.
    """
    results: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []

    active_types = _DETECTION_ORDER
    if entities:
        active_types = [t for t in _DETECTION_ORDER if t in entities]

    for entity_type in active_types:
        pattern = _PII_PATTERNS.get(entity_type)
        if pattern is None:
            continue
        for match in pattern.finditer(text):
            value = match.group()
            start = match.start()
            end = match.end()

            # Skip if this range overlaps with an already-detected entity
            if _overlaps(start, end, occupied):
                continue

            # Extra validation for credit cards
            if entity_type == "credit_card":
                if not _luhn_check(value):
                    continue

            results.append(
                {
                    "type": entity_type,
                    "value": value,
                    "start": start,
                    "end": end,
                }
            )
            occupied.append((start, end))

    # Sort by position
    results.sort(key=lambda e: e["start"])
    return results


def _validate_path(path: str) -> str | None:
    """Return an error message if the path is blocked, else None."""
    real_path = os.path.realpath(path)
    for prefix in _BLOCKED_PREFIXES:
        if real_path.startswith(prefix):
            return f"Access denied: reading from '{prefix}' is not allowed"
    return None


@tool(name="detect_pii")
def detect_pii_tool(text: str, entities: list = None, threshold: float = 0.0) -> dict:
    """Detect PII entities in text.

    Scans text for personally identifiable information using
    regex patterns. Detects emails, phone numbers, credit cards
    (with Luhn validation), SSNs, IP addresses, dates of birth,
    and URLs with embedded credentials.

    Args:
        text: Text to scan for PII entities.
        entities: Optional list of entity types to detect.
            Options: email, phone, credit_card, ssn, ip_address,
            date_of_birth, url_with_credentials.
            Defaults to all types.
        threshold: Confidence threshold (unused for regex, reserved for API compatibility).
    """
    if not text:
        raise ValueError("Parameter 'text' is required")

    entities_filter: list[str] | None = entities
    found = detect_pii(text, entities_filter)
    return {"entities": found, "count": len(found)}


@tool()
def scan_file_pii(file_path: str, entities: list = None) -> dict:
    """Scan a file for PII entities.

    Reads a file and scans its content for personally identifiable
    information. Includes security checks to block access to
    sensitive system paths.

    Args:
        file_path: Path to the file to scan.
        entities: Optional list of entity types to detect.
    """
    if not file_path:
        raise ValueError("Parameter 'file_path' is required")

    error = _validate_path(file_path)
    if error:
        raise PermissionError(error)

    real_path = os.path.realpath(file_path)
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.path.isfile(real_path):
        raise ValueError(f"Not a file: {file_path}")

    try:
        with open(real_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError) as e:
        raise OSError(f"Failed to read file: {e}")

    entities_filter: list[str] | None = entities
    found = detect_pii(content, entities_filter)

    # Determine which lines contain PII
    lines_with_pii: list[int] = []
    lines = content.split("\n")
    for entity in found:
        # Find line number for this entity's start position
        pos = 0
        for line_num, line in enumerate(lines, start=1):
            line_end = pos + len(line)
            if pos <= entity["start"] <= line_end:
                if line_num not in lines_with_pii:
                    lines_with_pii.append(line_num)
                break
            pos = line_end + 1  # +1 for newline

    lines_with_pii.sort()

    return {
        "file_path": file_path,
        "entities": found,
        "count": len(found),
        "lines_with_pii": lines_with_pii,
    }
