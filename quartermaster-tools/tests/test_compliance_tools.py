"""Tests for compliance tools: risk_classifier, audit_log, read_audit_log, compliance_checklist."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from quartermaster_tools.builtin.compliance.audit_log import (
    audit_log,
    read_audit_log,
)
from quartermaster_tools.builtin.compliance.checklist import compliance_checklist
from quartermaster_tools.builtin.compliance.risk_classifier import risk_classifier


# ---------------------------------------------------------------------------
# risk_classifier
# ---------------------------------------------------------------------------


class TestRiskClassifierTool:
    def setup_method(self) -> None:
        self.tool = risk_classifier

    def test_unacceptable_subliminal(self) -> None:
        result = self.tool.run(
            system_description="Subliminal ad system",
            domain="other",
            uses_subliminal_techniques=True,
        )
        assert result.success
        assert result.data["risk_level"] == "UNACCEPTABLE"

    def test_unacceptable_biometrics_law_enforcement(self) -> None:
        result = self.tool.run(
            system_description="Facial recognition for police",
            domain="law_enforcement",
            uses_biometrics=True,
        )
        assert result.success
        assert result.data["risk_level"] == "UNACCEPTABLE"

    def test_unacceptable_vulnerable_groups(self) -> None:
        result = self.tool.run(
            system_description="Targeting elderly with manipulative ads",
            domain="other",
            targets_vulnerable_groups=True,
        )
        assert result.success
        assert result.data["risk_level"] == "UNACCEPTABLE"

    def test_high_risk_healthcare(self) -> None:
        result = self.tool.run(
            system_description="Medical diagnosis AI",
            domain="healthcare",
        )
        assert result.success
        assert result.data["risk_level"] == "HIGH"
        assert "Art. 9" in str(result.data["obligations"])

    def test_high_risk_education(self) -> None:
        result = self.tool.run(
            system_description="Exam grading system",
            domain="education",
        )
        assert result.success
        assert result.data["risk_level"] == "HIGH"

    def test_high_risk_employment(self) -> None:
        result = self.tool.run(
            system_description="Resume screening AI",
            domain="employment",
        )
        assert result.success
        assert result.data["risk_level"] == "HIGH"

    def test_high_risk_biometrics_non_law_enforcement(self) -> None:
        result = self.tool.run(
            system_description="Biometric access control",
            domain="other",
            uses_biometrics=True,
        )
        assert result.success
        assert result.data["risk_level"] == "HIGH"

    def test_minimal_risk(self) -> None:
        result = self.tool.run(
            system_description="Spam filter",
            domain="other",
        )
        assert result.success
        assert result.data["risk_level"] == "MINIMAL"

    def test_missing_description(self) -> None:
        result = self.tool.run(domain="other")
        assert not result.success

    def test_missing_domain(self) -> None:
        result = self.tool.run(system_description="test")
        assert not result.success

    def test_reasoning_present(self) -> None:
        result = self.tool.run(system_description="test", domain="healthcare")
        assert result.success
        assert len(result.data["reasoning"]) > 0

    def test_obligations_present(self) -> None:
        result = self.tool.run(system_description="test", domain="healthcare")
        assert result.success
        assert len(result.data["obligations"]) > 5

    def test_info(self) -> None:
        info = self.tool.info()
        assert info.name == "risk_classifier"
        assert info.version == "1.0.0"


# ---------------------------------------------------------------------------
# audit_log
# ---------------------------------------------------------------------------


class TestAuditLogTool:
    def setup_method(self) -> None:
        self.tool = audit_log
        self._tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        self._tmpfile.close()
        self.log_path = self._tmpfile.name

    def teardown_method(self) -> None:
        if os.path.exists(self.log_path):
            os.unlink(self.log_path)

    def test_write_entry(self) -> None:
        result = self.tool.run(
            action="deploy",
            actor="admin",
            system_id="sys-001",
            log_path=self.log_path,
        )
        assert result.success
        assert result.data["logged"] is True
        assert result.data["entry_id"] == 0

    def test_write_multiple_entries(self) -> None:
        for i in range(3):
            result = self.tool.run(
                action=f"action_{i}",
                actor="admin",
                system_id="sys-001",
                log_path=self.log_path,
            )
            assert result.success
            assert result.data["entry_id"] == i

    def test_hash_chain(self) -> None:
        self.tool.run(action="a1", actor="admin", system_id="sys-001", log_path=self.log_path)
        self.tool.run(action="a2", actor="admin", system_id="sys-001", log_path=self.log_path)

        with open(self.log_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        entry0 = json.loads(lines[0])
        entry1 = json.loads(lines[1])

        # First entry should reference genesis hash
        assert entry0["previous_hash"] == "0" * 64

        # Second entry should hash the first line
        import hashlib

        expected = hashlib.sha256(lines[0].encode("utf-8")).hexdigest()
        assert entry1["previous_hash"] == expected

    def test_with_details(self) -> None:
        result = self.tool.run(
            action="update",
            actor="admin",
            system_id="sys-001",
            details={"version": "2.0"},
            log_path=self.log_path,
        )
        assert result.success
        with open(self.log_path) as f:
            entry = json.loads(f.readline())
        assert entry["details"]["version"] == "2.0"

    def test_missing_action(self) -> None:
        result = self.tool.run(actor="admin", system_id="sys-001")
        assert not result.success

    def test_info(self) -> None:
        info = self.tool.info()
        assert info.name == "audit_log"


# ---------------------------------------------------------------------------
# read_audit_log
# ---------------------------------------------------------------------------


class TestReadAuditLogTool:
    def setup_method(self) -> None:
        self.write_tool = audit_log
        self.read_tool = read_audit_log
        self._tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        self._tmpfile.close()
        self.log_path = self._tmpfile.name

    def teardown_method(self) -> None:
        if os.path.exists(self.log_path):
            os.unlink(self.log_path)

    def _write(self, action: str = "test", system_id: str = "sys-001") -> None:
        self.write_tool.run(
            action=action,
            actor="admin",
            system_id=system_id,
            log_path=self.log_path,
        )

    def test_read_entries(self) -> None:
        self._write()
        self._write()
        result = self.read_tool.run(system_id="sys-001", log_path=self.log_path)
        assert result.success
        assert result.data["count"] == 2

    def test_filter_by_system_id(self) -> None:
        self._write(system_id="sys-001")
        self._write(system_id="sys-002")
        result = self.read_tool.run(system_id="sys-001", log_path=self.log_path)
        assert result.success
        assert result.data["count"] == 1

    def test_filter_by_action(self) -> None:
        self._write(action="deploy")
        self._write(action="update")
        result = self.read_tool.run(
            system_id="sys-001",
            log_path=self.log_path,
            action_filter="deploy",
        )
        assert result.success
        assert result.data["count"] == 1
        assert result.data["entries"][0]["action"] == "deploy"

    def test_verify_integrity_valid(self) -> None:
        self._write()
        self._write()
        result = self.read_tool.run(
            system_id="sys-001",
            log_path=self.log_path,
            verify_integrity=True,
        )
        assert result.success
        assert result.data["integrity_valid"] is True

    def test_verify_integrity_tampered(self) -> None:
        self._write()
        self._write()
        # Tamper with the first line
        with open(self.log_path) as f:
            lines = f.readlines()
        entry = json.loads(lines[0])
        entry["action"] = "tampered"
        lines[0] = json.dumps(entry, separators=(",", ":")) + "\n"
        with open(self.log_path, "w") as f:
            f.writelines(lines)

        result = self.read_tool.run(
            system_id="sys-001",
            log_path=self.log_path,
            verify_integrity=True,
        )
        assert result.success
        assert result.data["integrity_valid"] is False

    def test_empty_log(self) -> None:
        result = self.read_tool.run(
            system_id="sys-001",
            log_path="/tmp/nonexistent_audit_log_test.jsonl",
        )
        assert result.success
        assert result.data["count"] == 0

    def test_missing_system_id(self) -> None:
        result = self.read_tool.run(log_path=self.log_path)
        assert not result.success

    def test_info(self) -> None:
        info = self.read_tool.info()
        assert info.name == "read_audit_log"


# ---------------------------------------------------------------------------
# compliance_checklist
# ---------------------------------------------------------------------------


class TestComplianceChecklistTool:
    def setup_method(self) -> None:
        self.tool = compliance_checklist

    def test_high_checklist(self) -> None:
        result = self.tool.run(risk_level="HIGH")
        assert result.success
        assert result.data["risk_level"] == "HIGH"
        assert result.data["total_items"] >= 10
        articles = [item["article"] for item in result.data["checklist"]]
        assert "Art. 9" in articles
        assert "Art. 10" in articles
        assert "Art. 14" in articles

    def test_unacceptable_checklist(self) -> None:
        result = self.tool.run(risk_level="UNACCEPTABLE")
        assert result.success
        assert result.data["total_items"] >= 4

    def test_limited_checklist(self) -> None:
        result = self.tool.run(risk_level="LIMITED")
        assert result.success
        assert result.data["total_items"] >= 3

    def test_minimal_checklist(self) -> None:
        result = self.tool.run(risk_level="MINIMAL")
        assert result.success
        assert result.data["total_items"] >= 2

    def test_case_insensitive(self) -> None:
        result = self.tool.run(risk_level="high")
        assert result.success
        assert result.data["risk_level"] == "HIGH"

    def test_invalid_risk_level(self) -> None:
        result = self.tool.run(risk_level="EXTREME")
        assert not result.success

    def test_missing_risk_level(self) -> None:
        result = self.tool.run()
        assert not result.success

    def test_checklist_items_have_required_keys(self) -> None:
        result = self.tool.run(risk_level="HIGH")
        for item in result.data["checklist"]:
            assert "article" in item
            assert "requirement" in item
            assert "status" in item
            assert item["status"] == "pending"

    def test_info(self) -> None:
        info = self.tool.info()
        assert info.name == "compliance_checklist"
