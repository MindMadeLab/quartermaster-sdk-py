"""
Email tools: send, read, and search via SMTP/IMAP.

Uses only stdlib modules (smtplib, imaplib, email).
"""

from __future__ import annotations

import email
import email.utils
import imaplib
import os
import smtplib
import time
from email.mime.text import MIMEText
from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

# Rate limiting for SendEmailTool
_send_timestamps: list[float] = []
_RATE_LIMIT = 10  # max sends per minute
_RATE_WINDOW = 60  # seconds


def _check_rate_limit() -> bool:
    """Return True if sending is allowed under rate limit."""
    now = time.time()
    # Remove timestamps older than the window
    while _send_timestamps and _send_timestamps[0] < now - _RATE_WINDOW:
        _send_timestamps.pop(0)
    return len(_send_timestamps) < _RATE_LIMIT


def _record_send() -> None:
    """Record a send timestamp for rate limiting."""
    _send_timestamps.append(time.time())


class SendEmailTool(AbstractTool):
    """Send an email via SMTP.

    Connection parameters fall back to SMTP_HOST, SMTP_PORT, SMTP_USER,
    SMTP_PASSWORD environment variables if not provided.
    Rate limited to 10 sends per minute.
    """

    def name(self) -> str:
        return "send_email"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="to", description="Recipient email address.", type="string", required=True),
            ToolParameter(name="subject", description="Email subject line.", type="string", required=True),
            ToolParameter(name="body", description="Email body text.", type="string", required=True),
            ToolParameter(name="cc", description="CC recipients (comma-separated).", type="string", required=False),
            ToolParameter(name="bcc", description="BCC recipients (comma-separated).", type="string", required=False),
            ToolParameter(name="smtp_host", description="SMTP server host (env: SMTP_HOST).", type="string", required=False),
            ToolParameter(name="smtp_port", description="SMTP server port (env: SMTP_PORT, default 587).", type="number", required=False, default=587),
            ToolParameter(name="smtp_user", description="SMTP username (env: SMTP_USER).", type="string", required=False),
            ToolParameter(name="smtp_password", description="SMTP password (env: SMTP_PASSWORD).", type="string", required=False),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Send an email via SMTP.",
            long_description=(
                "Sends an email using SMTP with TLS. Supports to, cc, bcc, "
                "subject, and body. Connection settings fall back to environment "
                "variables. Rate limited to 10 sends per minute."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        to_addr: str = kwargs.get("to", "").strip()
        subject: str = kwargs.get("subject", "").strip()
        body: str = kwargs.get("body", "")
        cc: str = kwargs.get("cc", "") or ""
        bcc: str = kwargs.get("bcc", "") or ""

        smtp_host = kwargs.get("smtp_host") or os.environ.get("SMTP_HOST", "")
        smtp_port = int(kwargs.get("smtp_port") or os.environ.get("SMTP_PORT", "587"))
        smtp_user = kwargs.get("smtp_user") or os.environ.get("SMTP_USER", "")
        smtp_password = kwargs.get("smtp_password") or os.environ.get("SMTP_PASSWORD", "")

        if not to_addr:
            return ToolResult(success=False, error="Parameter 'to' is required.")
        if not subject:
            return ToolResult(success=False, error="Parameter 'subject' is required.")
        if not body:
            return ToolResult(success=False, error="Parameter 'body' is required.")
        if not smtp_host:
            return ToolResult(
                success=False,
                error="SMTP host not provided. Set smtp_host parameter or SMTP_HOST env var.",
            )

        if not _check_rate_limit():
            return ToolResult(
                success=False,
                error="Rate limit exceeded: maximum 10 emails per minute.",
            )

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_addr
        if cc:
            msg["Cc"] = cc

        all_recipients = [to_addr]
        if cc:
            all_recipients.extend([a.strip() for a in cc.split(",") if a.strip()])
        if bcc:
            all_recipients.extend([a.strip() for a in bcc.split(",") if a.strip()])

        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user or to_addr, all_recipients, msg.as_string())
            _record_send()
            return ToolResult(
                success=True,
                data={"message": f"Email sent to {to_addr}", "to": to_addr, "subject": subject},
            )
        except smtplib.SMTPAuthenticationError as e:
            return ToolResult(success=False, error=f"SMTP authentication failed: {e}")
        except smtplib.SMTPException as e:
            return ToolResult(success=False, error=f"SMTP error: {e}")
        except OSError as e:
            return ToolResult(success=False, error=f"Connection error: {e}")


class ReadEmailTool(AbstractTool):
    """Read emails from an IMAP mailbox.

    Connection parameters fall back to IMAP_HOST, IMAP_USER, IMAP_PASSWORD
    environment variables if not provided.
    """

    def name(self) -> str:
        return "read_email"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="folder", description="Mailbox folder (default INBOX).", type="string", required=False, default="INBOX"),
            ToolParameter(name="count", description="Number of emails to fetch (default 10).", type="number", required=False, default=10),
            ToolParameter(name="unread_only", description="Only fetch unread emails (default True).", type="boolean", required=False, default=True),
            ToolParameter(name="imap_host", description="IMAP server host (env: IMAP_HOST).", type="string", required=False),
            ToolParameter(name="imap_user", description="IMAP username (env: IMAP_USER).", type="string", required=False),
            ToolParameter(name="imap_password", description="IMAP password (env: IMAP_PASSWORD).", type="string", required=False),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Read emails from an IMAP mailbox.",
            long_description=(
                "Connects to an IMAP server and fetches emails from the specified "
                "folder. Supports filtering by unread status. Connection settings "
                "fall back to environment variables."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        folder: str = kwargs.get("folder", "INBOX") or "INBOX"
        count: int = int(kwargs.get("count", 10) or 10)
        unread_only: bool = kwargs.get("unread_only", True)
        if isinstance(unread_only, str):
            unread_only = unread_only.lower() != "false"

        imap_host = kwargs.get("imap_host") or os.environ.get("IMAP_HOST", "")
        imap_user = kwargs.get("imap_user") or os.environ.get("IMAP_USER", "")
        imap_password = kwargs.get("imap_password") or os.environ.get("IMAP_PASSWORD", "")

        if not imap_host:
            return ToolResult(
                success=False,
                error="IMAP host not provided. Set imap_host parameter or IMAP_HOST env var.",
            )

        try:
            conn = imaplib.IMAP4_SSL(imap_host)
            conn.login(imap_user, imap_password)
            conn.select(folder, readonly=True)

            search_criteria = "(UNSEEN)" if unread_only else "ALL"
            _status, message_ids = conn.search(None, search_criteria)
            ids = message_ids[0].split() if message_ids[0] else []
            ids = ids[-count:]  # Take the most recent

            emails = []
            for msg_id in ids:
                _status, msg_data = conn.fetch(msg_id, "(RFC822)")
                if msg_data and msg_data[0] and isinstance(msg_data[0], tuple):
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                payload = part.get_payload(decode=True)
                                if payload:
                                    body = payload.decode("utf-8", errors="replace")
                                break
                    else:
                        payload = msg.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="replace")
                    emails.append({
                        "id": msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id),
                        "from": msg.get("From", ""),
                        "to": msg.get("To", ""),
                        "subject": msg.get("Subject", ""),
                        "date": msg.get("Date", ""),
                        "body": body[:2000],  # Truncate long bodies
                    })

            conn.close()
            conn.logout()

            return ToolResult(
                success=True,
                data={"folder": folder, "emails": emails, "email_count": len(emails)},
            )
        except imaplib.IMAP4.error as e:
            return ToolResult(success=False, error=f"IMAP error: {e}")
        except OSError as e:
            return ToolResult(success=False, error=f"Connection error: {e}")


class SearchEmailTool(AbstractTool):
    """Search emails via IMAP SEARCH command.

    Connection parameters fall back to IMAP_HOST, IMAP_USER, IMAP_PASSWORD
    environment variables if not provided.
    """

    def name(self) -> str:
        return "search_email"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="query", description="Search text (searches subject and body).", type="string", required=True),
            ToolParameter(name="folder", description="Mailbox folder (default INBOX).", type="string", required=False, default="INBOX"),
            ToolParameter(name="date_from", description="Search from date (DD-Mon-YYYY, e.g. 01-Jan-2024).", type="string", required=False),
            ToolParameter(name="date_to", description="Search to date (DD-Mon-YYYY, e.g. 31-Dec-2024).", type="string", required=False),
            ToolParameter(name="sender", description="Filter by sender email address.", type="string", required=False),
            ToolParameter(name="imap_host", description="IMAP server host (env: IMAP_HOST).", type="string", required=False),
            ToolParameter(name="imap_user", description="IMAP username (env: IMAP_USER).", type="string", required=False),
            ToolParameter(name="imap_password", description="IMAP password (env: IMAP_PASSWORD).", type="string", required=False),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Search emails via IMAP.",
            long_description=(
                "Searches emails using the IMAP SEARCH command. Supports filtering "
                "by text query, date range, and sender. Connection settings fall "
                "back to environment variables."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=False,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs.get("query", "").strip()
        folder: str = kwargs.get("folder", "INBOX") or "INBOX"
        date_from: str | None = kwargs.get("date_from")
        date_to: str | None = kwargs.get("date_to")
        sender: str | None = kwargs.get("sender")

        imap_host = kwargs.get("imap_host") or os.environ.get("IMAP_HOST", "")
        imap_user = kwargs.get("imap_user") or os.environ.get("IMAP_USER", "")
        imap_password = kwargs.get("imap_password") or os.environ.get("IMAP_PASSWORD", "")

        if not query:
            return ToolResult(success=False, error="Parameter 'query' is required.")
        if not imap_host:
            return ToolResult(
                success=False,
                error="IMAP host not provided. Set imap_host parameter or IMAP_HOST env var.",
            )

        # Build IMAP search criteria
        criteria_parts: list[str] = []
        criteria_parts.append(f'SUBJECT "{query}"')
        if date_from:
            criteria_parts.append(f'SINCE {date_from}')
        if date_to:
            criteria_parts.append(f'BEFORE {date_to}')
        if sender:
            criteria_parts.append(f'FROM "{sender}"')

        search_string = "(" + " ".join(criteria_parts) + ")"

        try:
            conn = imaplib.IMAP4_SSL(imap_host)
            conn.login(imap_user, imap_password)
            conn.select(folder, readonly=True)

            _status, message_ids = conn.search(None, search_string)
            ids = message_ids[0].split() if message_ids[0] else []

            emails = []
            for msg_id in ids[-50:]:  # Limit to 50 results
                _status, msg_data = conn.fetch(msg_id, "(RFC822.HEADER)")
                if msg_data and msg_data[0] and isinstance(msg_data[0], tuple):
                    raw_header = msg_data[0][1]
                    msg = email.message_from_bytes(raw_header)
                    emails.append({
                        "id": msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id),
                        "from": msg.get("From", ""),
                        "to": msg.get("To", ""),
                        "subject": msg.get("Subject", ""),
                        "date": msg.get("Date", ""),
                    })

            conn.close()
            conn.logout()

            return ToolResult(
                success=True,
                data={"query": query, "folder": folder, "emails": emails, "email_count": len(emails)},
            )
        except imaplib.IMAP4.error as e:
            return ToolResult(success=False, error=f"IMAP error: {e}")
        except OSError as e:
            return ToolResult(success=False, error=f"Connection error: {e}")
