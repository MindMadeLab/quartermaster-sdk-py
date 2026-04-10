"""
RedactPIITool: Redact PII from text with configurable strategies.

Strategies:
- redact: Replace with type labels like <EMAIL>, <PHONE>
- mask: Partially mask values, keeping first/last chars
- hash: Replace with first 8 chars of SHA-256 hash
"""

from __future__ import annotations

import hashlib
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.builtin.privacy.detect import detect_pii
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

# Labels for redaction
_REDACT_LABELS: dict[str, str] = {
    "email": "<EMAIL>",
    "phone": "<PHONE>",
    "credit_card": "<CREDIT_CARD>",
    "ssn": "<SSN>",
    "ip_address": "<IP_ADDRESS>",
    "date_of_birth": "<DATE_OF_BIRTH>",
    "url_with_credentials": "<URL_WITH_CREDENTIALS>",
}


def _mask_value(entity_type: str, value: str) -> str:
    """Mask a PII value, keeping first and last chars of segments."""
    if entity_type == "email":
        parts = value.split("@")
        if len(parts) == 2:
            local = parts[0]
            domain = parts[1]
            masked_local = local[0] + "***" if len(local) > 1 else local
            domain_parts = domain.split(".")
            masked_domain = ".".join(
                p[0] + "*" * (len(p) - 1) if len(p) > 1 else p
                for p in domain_parts
            )
            return f"{masked_local}@{masked_domain}"
    if entity_type == "phone":
        digits = [c for c in value if c.isdigit()]
        if len(digits) >= 4:
            return value[:2] + "*" * (len(value) - 4) + value[-2:]
    if entity_type == "credit_card":
        digits_only = "".join(c for c in value if c.isdigit())
        if len(digits_only) >= 8:
            return digits_only[:4] + "*" * (len(digits_only) - 8) + digits_only[-4:]
    if entity_type == "ssn":
        return "***-**-" + value[-4:]
    if entity_type == "ip_address":
        parts = value.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.***.***.{parts[3]}"
    # Default: keep first and last char
    if len(value) > 2:
        return value[0] + "*" * (len(value) - 2) + value[-1]
    return "*" * len(value)


def _hash_value(value: str) -> str:
    """Return first 8 chars of SHA-256 hash of the value."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def redact_text(
    text: str,
    strategy: str = "redact",
    entities_filter: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Redact PII from text.

    Args:
        text: The text to redact.
        strategy: One of 'redact', 'mask', 'hash'.
        entities_filter: Optional entity types to filter.

    Returns:
        Tuple of (redacted_text, entities_found).
    """
    found = detect_pii(text, entities_filter)
    if not found:
        return text, found

    # Process replacements from end to start to preserve positions
    result = text
    for entity in reversed(found):
        entity_type = entity["type"]
        value = entity["value"]
        start = entity["start"]
        end = entity["end"]

        if strategy == "redact":
            replacement = _REDACT_LABELS.get(entity_type, f"<{entity_type.upper()}>")
        elif strategy == "mask":
            replacement = _mask_value(entity_type, value)
        elif strategy == "hash":
            replacement = _hash_value(value)
        else:
            replacement = _REDACT_LABELS.get(entity_type, f"<{entity_type.upper()}>")

        result = result[:start] + replacement + result[end:]

    return result, found


class RedactPIITool(AbstractTool):
    """Redact PII from text with configurable strategy."""

    def name(self) -> str:
        return "redact_pii"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                description="Text from which to redact PII.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="strategy",
                description="Redaction strategy: 'redact', 'mask', or 'hash'.",
                type="string",
                required=False,
                default="redact",
            ),
            ToolParameter(
                name="entities",
                description="Optional list of entity types to redact.",
                type="array",
                required=False,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Redact PII from text.",
            long_description=(
                "Removes or masks personally identifiable information from text. "
                "Supports three strategies: redact (replace with labels), "
                "mask (partial masking), and hash (SHA-256 based replacement)."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        text: str = kwargs.get("text", "")
        if not text:
            return ToolResult(success=False, error="Parameter 'text' is required")

        strategy: str = kwargs.get("strategy", "redact")
        if strategy not in ("redact", "mask", "hash"):
            return ToolResult(
                success=False,
                error=f"Invalid strategy: {strategy!r}. Must be 'redact', 'mask', or 'hash'.",
            )

        entities_filter: list[str] | None = kwargs.get("entities")
        redacted, found = redact_text(text, strategy, entities_filter)

        return ToolResult(
            success=True,
            data={
                "redacted_text": redacted,
                "entities_found": len(found),
                "entities": found,
            },
        )
