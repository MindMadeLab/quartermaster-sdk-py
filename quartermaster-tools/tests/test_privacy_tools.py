"""Tests for privacy tools: DetectPIITool, RedactPIITool, ScanFilePIITool."""

from __future__ import annotations

import os
import tempfile

import pytest

from quartermaster_tools.builtin.privacy.detect import DetectPIITool, ScanFilePIITool
from quartermaster_tools.builtin.privacy.redact import RedactPIITool


# ---------------------------------------------------------------------------
# DetectPIITool
# ---------------------------------------------------------------------------

class TestDetectPIITool:
    def setup_method(self) -> None:
        self.tool = DetectPIITool

    def test_detect_email(self) -> None:
        result = self.tool.run(text="Contact john@example.com for details.")
        assert result.success
        entities = result.data["entities"]
        assert len(entities) == 1
        assert entities[0]["type"] == "email"
        assert entities[0]["value"] == "john@example.com"

    def test_detect_multiple_emails(self) -> None:
        result = self.tool.run(text="Send to a@b.com and c@d.org please.")
        assert result.success
        emails = [e for e in result.data["entities"] if e["type"] == "email"]
        assert len(emails) == 2

    def test_detect_phone_us(self) -> None:
        result = self.tool.run(text="Call me at +1 (555) 123-4567.")
        assert result.success
        phones = [e for e in result.data["entities"] if e["type"] == "phone"]
        assert len(phones) >= 1

    def test_detect_credit_card_visa(self) -> None:
        # Valid Visa test number
        result = self.tool.run(text="Card: 4111 1111 1111 1111")
        assert result.success
        cards = [e for e in result.data["entities"] if e["type"] == "credit_card"]
        assert len(cards) == 1

    def test_reject_invalid_credit_card(self) -> None:
        # Invalid Luhn
        result = self.tool.run(text="Card: 4111 1111 1111 1112")
        assert result.success
        cards = [e for e in result.data["entities"] if e["type"] == "credit_card"]
        assert len(cards) == 0

    def test_detect_ssn(self) -> None:
        result = self.tool.run(text="SSN: 123-45-6789")
        assert result.success
        ssns = [e for e in result.data["entities"] if e["type"] == "ssn"]
        assert len(ssns) == 1
        assert ssns[0]["value"] == "123-45-6789"

    def test_detect_ip_address(self) -> None:
        result = self.tool.run(text="Server IP: 192.168.1.100")
        assert result.success
        ips = [e for e in result.data["entities"] if e["type"] == "ip_address"]
        assert len(ips) == 1

    def test_detect_date_of_birth(self) -> None:
        result = self.tool.run(text="DOB: 01/15/1990")
        assert result.success
        dobs = [e for e in result.data["entities"] if e["type"] == "date_of_birth"]
        assert len(dobs) == 1

    def test_detect_url_with_credentials(self) -> None:
        result = self.tool.run(text="URL: https://user:pass@example.com/path")
        assert result.success
        urls = [e for e in result.data["entities"] if e["type"] == "url_with_credentials"]
        assert len(urls) == 1

    def test_no_pii(self) -> None:
        result = self.tool.run(text="The quick brown fox jumps over the lazy dog.")
        assert result.success
        assert result.data["count"] == 0
        assert result.data["entities"] == []

    def test_entity_filtering(self) -> None:
        text = "Email: test@example.com, SSN: 123-45-6789"
        result = self.tool.run(text=text, entities=["email"])
        assert result.success
        assert all(e["type"] == "email" for e in result.data["entities"])
        assert result.data["count"] == 1

    def test_mixed_text(self) -> None:
        text = (
            "Contact john@example.com or call +1-555-123-4567. "
            "SSN: 123-45-6789. IP: 10.0.0.1"
        )
        result = self.tool.run(text=text)
        assert result.success
        assert result.data["count"] >= 3

    def test_positions(self) -> None:
        text = "Email: test@example.com"
        result = self.tool.run(text=text)
        entity = result.data["entities"][0]
        assert text[entity["start"]:entity["end"]] == entity["value"]

    def test_missing_text(self) -> None:
        result = self.tool.run()
        assert not result.success
        assert "required" in result.error.lower()

    def test_info(self) -> None:
        info = self.tool.info()
        assert info.name == "detect_pii"
        assert info.version == "1.0.0"


# ---------------------------------------------------------------------------
# RedactPIITool
# ---------------------------------------------------------------------------

class TestRedactPIITool:
    def setup_method(self) -> None:
        self.tool = RedactPIITool

    def test_redact_strategy(self) -> None:
        text = "Email: john@example.com"
        result = self.tool.run(text=text, strategy="redact")
        assert result.success
        assert "<EMAIL>" in result.data["redacted_text"]
        assert "john@example.com" not in result.data["redacted_text"]

    def test_mask_strategy_email(self) -> None:
        text = "Email: john@example.com"
        result = self.tool.run(text=text, strategy="mask")
        assert result.success
        redacted = result.data["redacted_text"]
        assert "john@example.com" not in redacted
        # Should keep first char of local part
        assert "j***@" in redacted

    def test_hash_strategy(self) -> None:
        text = "Email: john@example.com"
        result = self.tool.run(text=text, strategy="hash")
        assert result.success
        redacted = result.data["redacted_text"]
        assert "john@example.com" not in redacted
        # Hash is 8 hex chars
        assert len(redacted.replace("Email: ", "")) == 8

    def test_redact_ssn(self) -> None:
        text = "SSN: 123-45-6789"
        result = self.tool.run(text=text, strategy="redact")
        assert result.success
        assert "<SSN>" in result.data["redacted_text"]

    def test_redact_credit_card(self) -> None:
        text = "Card: 4111 1111 1111 1111"
        result = self.tool.run(text=text, strategy="redact")
        assert result.success
        assert "<CREDIT_CARD>" in result.data["redacted_text"]

    def test_entity_filtering_redact(self) -> None:
        text = "Email: test@example.com, SSN: 123-45-6789"
        result = self.tool.run(text=text, strategy="redact", entities=["email"])
        assert result.success
        assert "<EMAIL>" in result.data["redacted_text"]
        # SSN should remain
        assert "123-45-6789" in result.data["redacted_text"]

    def test_no_pii_redact(self) -> None:
        text = "Hello world"
        result = self.tool.run(text=text)
        assert result.success
        assert result.data["redacted_text"] == text
        assert result.data["entities_found"] == 0

    def test_invalid_strategy(self) -> None:
        result = self.tool.run(text="test", strategy="invalid")
        assert not result.success

    def test_mask_ssn(self) -> None:
        text = "SSN: 123-45-6789"
        result = self.tool.run(text=text, strategy="mask")
        assert result.success
        assert "6789" in result.data["redacted_text"]
        assert "123-45-6789" not in result.data["redacted_text"]

    def test_info(self) -> None:
        info = self.tool.info()
        assert info.name == "redact_pii"


# ---------------------------------------------------------------------------
# ScanFilePIITool
# ---------------------------------------------------------------------------

class TestScanFilePIITool:
    def setup_method(self) -> None:
        self.tool = ScanFilePIITool

    def test_scan_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Line 1: no pii\n")
            f.write("Line 2: email is test@example.com\n")
            f.write("Line 3: no pii\n")
            path = f.name

        try:
            result = self.tool.run(file_path=path)
            assert result.success
            assert result.data["count"] == 1
            assert 2 in result.data["lines_with_pii"]
        finally:
            os.unlink(path)

    def test_blocked_path_etc(self) -> None:
        result = self.tool.run(file_path="/etc/passwd")
        assert not result.success
        assert "denied" in result.error.lower() or "not allowed" in result.error.lower()

    def test_blocked_path_proc(self) -> None:
        result = self.tool.run(file_path="/proc/self/environ")
        assert not result.success

    def test_file_not_found(self) -> None:
        result = self.tool.run(file_path="/tmp/nonexistent_pii_test_file.txt")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_missing_file_path(self) -> None:
        result = self.tool.run()
        assert not result.success

    def test_entity_filtering_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Email: a@b.com\nSSN: 123-45-6789\n")
            path = f.name

        try:
            result = self.tool.run(file_path=path, entities=["ssn"])
            assert result.success
            types = [e["type"] for e in result.data["entities"]]
            assert all(t == "ssn" for t in types)
        finally:
            os.unlink(path)

    def test_info(self) -> None:
        info = self.tool.info()
        assert info.name == "scan_file_pii"
