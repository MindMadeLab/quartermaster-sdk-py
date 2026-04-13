"""
Email tools: send, read, and search via SMTP/IMAP.

Uses only stdlib modules (smtplib, imaplib, email).
"""

from __future__ import annotations

import email as email_lib
import email.utils
import imaplib
import os
import smtplib
import time
from email.mime.text import MIMEText
from typing import Any

from quartermaster_tools.decorator import tool
from quartermaster_tools.types import ToolResult

# Rate limiting for send_email
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


@tool()
def send_email(
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    bcc: str = "",
    smtp_host: str = "",
    smtp_port: int = 587,
    smtp_user: str = "",
    smtp_password: str = "",
) -> ToolResult:
    """Send an email via SMTP.

    Sends an email using SMTP with TLS. Supports to, cc, bcc, subject, and body.
    Connection settings fall back to environment variables. Rate limited to 10
    sends per minute.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.
        cc: CC recipients (comma-separated).
        bcc: BCC recipients (comma-separated).
        smtp_host: SMTP server host (env: SMTP_HOST).
        smtp_port: SMTP server port (env: SMTP_PORT, default 587).
        smtp_user: SMTP username (env: SMTP_USER).
        smtp_password: SMTP password (env: SMTP_PASSWORD).
    """
    to_addr = (to or "").strip()
    subject_val = (subject or "").strip()
    body_val = body or ""
    cc_val = cc or ""
    bcc_val = bcc or ""

    host = smtp_host or os.environ.get("SMTP_HOST", "")
    port = int(smtp_port or os.environ.get("SMTP_PORT", "587"))
    user = smtp_user or os.environ.get("SMTP_USER", "")
    password = smtp_password or os.environ.get("SMTP_PASSWORD", "")

    if not to_addr:
        return ToolResult(success=False, error="Parameter 'to' is required.")
    if not subject_val:
        return ToolResult(success=False, error="Parameter 'subject' is required.")
    if not body_val:
        return ToolResult(success=False, error="Parameter 'body' is required.")
    if not host:
        return ToolResult(
            success=False,
            error="SMTP host not provided. Set smtp_host parameter or SMTP_HOST env var.",
        )

    if not _check_rate_limit():
        return ToolResult(
            success=False,
            error="Rate limit exceeded: maximum 10 emails per minute.",
        )

    msg = MIMEText(body_val)
    msg["Subject"] = subject_val
    msg["From"] = user
    msg["To"] = to_addr
    if cc_val:
        msg["Cc"] = cc_val

    all_recipients = [to_addr]
    if cc_val:
        all_recipients.extend([a.strip() for a in cc_val.split(",") if a.strip()])
    if bcc_val:
        all_recipients.extend([a.strip() for a in bcc_val.split(",") if a.strip()])

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            if user and password:
                server.login(user, password)
            server.sendmail(user or to_addr, all_recipients, msg.as_string())
        _record_send()
        return ToolResult(
            success=True,
            data={"message": f"Email sent to {to_addr}", "to": to_addr, "subject": subject_val},
        )
    except smtplib.SMTPAuthenticationError as e:
        return ToolResult(success=False, error=f"SMTP authentication failed: {e}")
    except smtplib.SMTPException as e:
        return ToolResult(success=False, error=f"SMTP error: {e}")
    except OSError as e:
        return ToolResult(success=False, error=f"Connection error: {e}")


@tool()
def read_email(
    folder: str = "INBOX",
    count: int = 10,
    unread_only: bool = True,
    imap_host: str = "",
    imap_user: str = "",
    imap_password: str = "",
) -> ToolResult:
    """Read emails from an IMAP mailbox.

    Connects to an IMAP server and fetches emails from the specified folder.
    Supports filtering by unread status. Connection settings fall back to
    environment variables.

    Args:
        folder: Mailbox folder (default INBOX).
        count: Number of emails to fetch (default 10).
        unread_only: Only fetch unread emails (default True).
        imap_host: IMAP server host (env: IMAP_HOST).
        imap_user: IMAP username (env: IMAP_USER).
        imap_password: IMAP password (env: IMAP_PASSWORD).
    """
    folder_val = folder or "INBOX"
    count_val = int(count or 10)
    unread_val = unread_only
    if isinstance(unread_val, str):
        unread_val = unread_val.lower() != "false"

    host = imap_host or os.environ.get("IMAP_HOST", "")
    user = imap_user or os.environ.get("IMAP_USER", "")
    password = imap_password or os.environ.get("IMAP_PASSWORD", "")

    if not host:
        return ToolResult(
            success=False,
            error="IMAP host not provided. Set imap_host parameter or IMAP_HOST env var.",
        )

    try:
        conn = imaplib.IMAP4_SSL(host)
        conn.login(user, password)
        conn.select(folder_val, readonly=True)

        search_criteria = "(UNSEEN)" if unread_val else "ALL"
        _status, message_ids = conn.search(None, search_criteria)
        ids = message_ids[0].split() if message_ids[0] else []
        ids = ids[-count_val:]  # Take the most recent

        emails = []
        for msg_id in ids:
            _status, msg_data = conn.fetch(msg_id, "(RFC822)")
            if msg_data and msg_data[0] and isinstance(msg_data[0], tuple):
                raw_email = msg_data[0][1]
                msg = email_lib.message_from_bytes(raw_email)
                body_text = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                body_text = payload.decode("utf-8", errors="replace")
                            break
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body_text = payload.decode("utf-8", errors="replace")
                emails.append({
                    "id": msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id),
                    "from": msg.get("From", ""),
                    "to": msg.get("To", ""),
                    "subject": msg.get("Subject", ""),
                    "date": msg.get("Date", ""),
                    "body": body_text[:2000],
                })

        conn.close()
        conn.logout()

        return ToolResult(
            success=True,
            data={"folder": folder_val, "emails": emails, "email_count": len(emails)},
        )
    except imaplib.IMAP4.error as e:
        return ToolResult(success=False, error=f"IMAP error: {e}")
    except OSError as e:
        return ToolResult(success=False, error=f"Connection error: {e}")


@tool()
def search_email(
    query: str,
    folder: str = "INBOX",
    date_from: str = None,
    date_to: str = None,
    sender: str = None,
    imap_host: str = "",
    imap_user: str = "",
    imap_password: str = "",
) -> ToolResult:
    """Search emails via IMAP.

    Searches emails using the IMAP SEARCH command. Supports filtering by text
    query, date range, and sender. Connection settings fall back to environment
    variables.

    Args:
        query: Search text (searches subject and body).
        folder: Mailbox folder (default INBOX).
        date_from: Search from date (DD-Mon-YYYY, e.g. 01-Jan-2024).
        date_to: Search to date (DD-Mon-YYYY, e.g. 31-Dec-2024).
        sender: Filter by sender email address.
        imap_host: IMAP server host (env: IMAP_HOST).
        imap_user: IMAP username (env: IMAP_USER).
        imap_password: IMAP password (env: IMAP_PASSWORD).
    """
    query_val = (query or "").strip()
    folder_val = folder or "INBOX"

    host = imap_host or os.environ.get("IMAP_HOST", "")
    user = imap_user or os.environ.get("IMAP_USER", "")
    password = imap_password or os.environ.get("IMAP_PASSWORD", "")

    if not query_val:
        return ToolResult(success=False, error="Parameter 'query' is required.")
    if not host:
        return ToolResult(
            success=False,
            error="IMAP host not provided. Set imap_host parameter or IMAP_HOST env var.",
        )

    # Build IMAP search criteria
    criteria_parts: list[str] = []
    criteria_parts.append(f'SUBJECT "{query_val}"')
    if date_from:
        criteria_parts.append(f'SINCE {date_from}')
    if date_to:
        criteria_parts.append(f'BEFORE {date_to}')
    if sender:
        criteria_parts.append(f'FROM "{sender}"')

    search_string = "(" + " ".join(criteria_parts) + ")"

    try:
        conn = imaplib.IMAP4_SSL(host)
        conn.login(user, password)
        conn.select(folder_val, readonly=True)

        _status, message_ids = conn.search(None, search_string)
        ids = message_ids[0].split() if message_ids[0] else []

        emails = []
        for msg_id in ids[-50:]:
            _status, msg_data = conn.fetch(msg_id, "(RFC822.HEADER)")
            if msg_data and msg_data[0] and isinstance(msg_data[0], tuple):
                raw_header = msg_data[0][1]
                msg = email_lib.message_from_bytes(raw_header)
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
            data={"query": query_val, "folder": folder_val, "emails": emails, "email_count": len(emails)},
        )
    except imaplib.IMAP4.error as e:
        return ToolResult(success=False, error=f"IMAP error: {e}")
    except OSError as e:
        return ToolResult(success=False, error=f"Connection error: {e}")


# Backward-compatible aliases
SendEmailTool = send_email
ReadEmailTool = read_email
SearchEmailTool = search_email
