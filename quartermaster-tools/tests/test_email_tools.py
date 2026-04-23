"""Tests for send_email, read_email, and search_email tools."""

from __future__ import annotations

import imaplib
import smtplib
from unittest.mock import MagicMock, patch

import pytest

from quartermaster_tools.builtin.email.tools import (
    send_email,
    read_email,
    search_email,
    send_email,
    read_email,
    search_email,
    _send_timestamps,
)


class TestSendEmailTool:
    def setup_method(self) -> None:
        _send_timestamps.clear()

    def test_name(self) -> None:
        assert send_email.name() == "send_email"

    def test_version(self) -> None:
        assert send_email.version() == "1.0.0"

    def test_parameters_list(self) -> None:
        params = send_email.parameters()
        names = [p.name for p in params]
        assert "to" in names
        assert "subject" in names
        assert "body" in names
        assert "smtp_host" in names

    def test_info_descriptor(self) -> None:
        info = send_email.info()
        assert info.name == "send_email"

    def test_missing_to(self) -> None:
        result = send_email.run(to="", subject="hi", body="hello")
        assert result.success is False
        assert "to" in result.error.lower()

    def test_missing_subject(self) -> None:
        result = send_email.run(to="a@b.com", subject="", body="hello")
        assert result.success is False
        assert "subject" in result.error.lower()

    def test_missing_body(self) -> None:
        result = send_email.run(to="a@b.com", subject="hi", body="")
        assert result.success is False
        assert "body" in result.error.lower()

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_smtp_host(self) -> None:
        result = send_email.run(to="a@b.com", subject="hi", body="hello")
        assert result.success is False
        assert "smtp" in result.error.lower()

    @patch("quartermaster_tools.builtin.email.tools.smtplib.SMTP")
    @patch.dict(
        "os.environ",
        {"SMTP_HOST": "mail.example.com", "SMTP_USER": "user", "SMTP_PASSWORD": "pass"},
    )
    def test_successful_send(self, mock_smtp_cls: MagicMock) -> None:
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = send_email.run(to="recipient@example.com", subject="Test", body="Hello!")
        assert result.success is True
        assert "recipient@example.com" in result.data["to"]
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.sendmail.assert_called_once()

    @patch("quartermaster_tools.builtin.email.tools.smtplib.SMTP")
    @patch.dict(
        "os.environ",
        {"SMTP_HOST": "mail.example.com", "SMTP_USER": "user", "SMTP_PASSWORD": "pass"},
    )
    def test_send_with_cc_bcc(self, mock_smtp_cls: MagicMock) -> None:
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = send_email.run(
            to="a@b.com", subject="Test", body="Hello", cc="c@d.com", bcc="e@f.com"
        )
        assert result.success is True
        sendmail_args = mock_server.sendmail.call_args[0]
        recipients = sendmail_args[1]
        assert "a@b.com" in recipients
        assert "c@d.com" in recipients
        assert "e@f.com" in recipients

    @patch("quartermaster_tools.builtin.email.tools.smtplib.SMTP")
    @patch.dict("os.environ", {"SMTP_HOST": "mail.example.com"})
    def test_smtp_auth_error(self, mock_smtp_cls: MagicMock) -> None:
        mock_server = MagicMock()
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Auth failed")
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = send_email.run(
            to="a@b.com", subject="Test", body="Hello", smtp_user="bad", smtp_password="creds"
        )
        assert result.success is False
        assert "authentication" in result.error.lower()

    @patch.dict("os.environ", {"SMTP_HOST": "mail.example.com"})
    def test_rate_limit(self) -> None:
        import time

        # Fill up rate limit
        now = time.time()
        _send_timestamps.clear()
        for _ in range(10):
            _send_timestamps.append(now)

        result = send_email.run(to="a@b.com", subject="Test", body="Hello")
        assert result.success is False
        assert "rate limit" in result.error.lower()


class TestReadEmailTool:
    def test_name(self) -> None:
        assert read_email.name() == "read_email"

    def test_version(self) -> None:
        assert read_email.version() == "1.0.0"

    def test_info_descriptor(self) -> None:
        info = read_email.info()
        assert info.name == "read_email"

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_imap_host(self) -> None:
        result = read_email.run()
        assert result.success is False
        assert "imap" in result.error.lower()

    @patch("quartermaster_tools.builtin.email.tools.imaplib.IMAP4_SSL")
    @patch.dict(
        "os.environ",
        {"IMAP_HOST": "imap.example.com", "IMAP_USER": "user", "IMAP_PASSWORD": "pass"},
    )
    def test_successful_read(self, mock_imap_cls: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_imap_cls.return_value = mock_conn
        mock_conn.search.return_value = ("OK", [b"1 2"])
        # Build a minimal email
        raw_email = (
            b"From: sender@example.com\r\nTo: me@example.com\r\n"
            b"Subject: Test Email\r\nDate: Mon, 1 Jan 2024 00:00:00 +0000\r\n\r\n"
            b"Hello world"
        )
        mock_conn.fetch.return_value = ("OK", [(b"1", raw_email)])

        result = read_email.run()
        assert result.success is True
        assert result.data["folder"] == "INBOX"
        assert len(result.data["emails"]) == 2
        mock_conn.login.assert_called_once()
        mock_conn.close.assert_called_once()
        mock_conn.logout.assert_called_once()

    @patch("quartermaster_tools.builtin.email.tools.imaplib.IMAP4_SSL")
    @patch.dict(
        "os.environ",
        {"IMAP_HOST": "imap.example.com", "IMAP_USER": "user", "IMAP_PASSWORD": "pass"},
    )
    def test_imap_error(self, mock_imap_cls: MagicMock) -> None:
        mock_imap_cls.side_effect = imaplib.IMAP4.error("Login failed")
        result = read_email.run()
        assert result.success is False
        assert "imap" in result.error.lower()

    @patch("quartermaster_tools.builtin.email.tools.imaplib.IMAP4_SSL")
    @patch.dict(
        "os.environ",
        {"IMAP_HOST": "imap.example.com", "IMAP_USER": "user", "IMAP_PASSWORD": "pass"},
    )
    def test_read_all_not_unread_only(self, mock_imap_cls: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_imap_cls.return_value = mock_conn
        mock_conn.search.return_value = ("OK", [b""])

        result = read_email.run(unread_only=False)
        assert result.success is True
        mock_conn.search.assert_called_once_with(None, "ALL")


class TestSearchEmailTool:
    def test_name(self) -> None:
        assert search_email.name() == "search_email"

    def test_version(self) -> None:
        assert search_email.version() == "1.0.0"

    def test_info_descriptor(self) -> None:
        info = search_email.info()
        assert info.name == "search_email"

    def test_missing_query(self) -> None:
        result = search_email.run(query="")
        assert result.success is False
        assert "query" in result.error.lower()

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_imap_host(self) -> None:
        result = search_email.run(query="test")
        assert result.success is False
        assert "imap" in result.error.lower()

    @patch("quartermaster_tools.builtin.email.tools.imaplib.IMAP4_SSL")
    @patch.dict(
        "os.environ",
        {"IMAP_HOST": "imap.example.com", "IMAP_USER": "user", "IMAP_PASSWORD": "pass"},
    )
    def test_successful_search(self, mock_imap_cls: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_imap_cls.return_value = mock_conn
        mock_conn.search.return_value = ("OK", [b"1"])
        raw_header = (
            b"From: sender@example.com\r\nTo: me@example.com\r\n"
            b"Subject: Found Email\r\nDate: Mon, 1 Jan 2024 00:00:00 +0000\r\n\r\n"
        )
        mock_conn.fetch.return_value = ("OK", [(b"1", raw_header)])

        result = search_email.run(query="Found")
        assert result.success is True
        assert result.data["email_count"] == 1
        assert result.data["emails"][0]["subject"] == "Found Email"

    @patch("quartermaster_tools.builtin.email.tools.imaplib.IMAP4_SSL")
    @patch.dict(
        "os.environ",
        {"IMAP_HOST": "imap.example.com", "IMAP_USER": "user", "IMAP_PASSWORD": "pass"},
    )
    def test_search_with_date_and_sender(self, mock_imap_cls: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_imap_cls.return_value = mock_conn
        mock_conn.search.return_value = ("OK", [b""])

        result = search_email.run(
            query="test", date_from="01-Jan-2024", date_to="31-Dec-2024", sender="boss@co.com"
        )
        assert result.success is True
        search_arg = mock_conn.search.call_args[0][1]
        assert "SINCE 01-Jan-2024" in search_arg
        assert "BEFORE 31-Dec-2024" in search_arg
        assert 'FROM "boss@co.com"' in search_arg


class TestEmailToolAliases:
    def test_backward_compat_aliases(self) -> None:
        assert send_email is send_email
        assert read_email is read_email
        assert search_email is search_email
