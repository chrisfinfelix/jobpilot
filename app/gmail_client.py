"""
Step 5 — Gmail send flow.

Handles:
  - Building the Google OAuth consent URL
  - Exchanging the callback code for tokens
  - Persisting/refreshing the token locally (single-user app, so a flat
    JSON file is enough — see SETUP_STEP5.md for why this doesn't scale
    to multi-user without changes)
  - Building and sending the actual email via the Gmail API

IMPORTANT: this module only ever sends an email when send_email() is
called explicitly with a final, user-approved subject/body/recipient.
Nothing in here decides on its own to send anything — that decision is
made by the human in the review-gate UI (see static/index.html Step 5
section). Do not wire send_email() to anything that fires automatically.
"""

import os
import json
import base64
import secrets
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Google's consent screen silently appends "openid", "userinfo.profile",
# and "userinfo.email" to whatever scopes you actually requested (this is
# normal Google behavior, not something we're asking for). Without this
# flag, oauthlib treats that as a scope mismatch and raises a Warning that
# surfaces to the user as "Gmail connection failed: Scope has changed...".
# This must be set before Flow.fetch_token() is called.
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Only scope requested: send-only. We deliberately do NOT request
# gmail.readonly / gmail.modify — the app never needs to read the
# user's mailbox, only send on explicit approval.
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

TOKEN_PATH = os.path.join(os.path.dirname(__file__), "data", "gmail_token.json")

# In-memory CSRF state store. Fine for a single-user local app; if this
# ever becomes multi-user/multi-process, move to the same SQLite cache
# used elsewhere (research_cache.py) with a namespace like "oauth_state".
_pending_states: set[str] = set()


def _client_config() -> dict:
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/gmail/callback")

    if not client_id or not client_secret:
        raise RuntimeError(
            "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET are not set. "
            "See SETUP_STEP5.md to create an OAuth client and add them to .env."
        )

    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }, redirect_uri


def build_auth_url() -> str:
    """Kick off the OAuth flow. Returns the URL the frontend should
    redirect the browser to for the Google consent screen."""
    config, redirect_uri = _client_config()
    flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=redirect_uri)

    state = secrets.token_urlsafe(24)
    _pending_states.add(state)

    auth_url, _ = flow.authorization_url(
        access_type="offline",       # needed to get a refresh token
        include_granted_scopes="true",
        prompt="consent",            # forces refresh token on every connect, not just the first
        state=state,
    )
    return auth_url


def handle_callback(code: str, state: str) -> None:
    """Exchange the authorization code for tokens and persist them.
    Raises ValueError on an unrecognized/replayed state param."""
    if state not in _pending_states:
        raise ValueError("Unrecognized or expired OAuth state — possible CSRF or stale link. Try connecting again.")
    _pending_states.discard(state)

    config, redirect_uri = _client_config()
    flow = Flow.from_client_config(config, scopes=SCOPES, redirect_uri=redirect_uri)
    flow.fetch_token(code=code)

    creds = flow.credentials
    _save_credentials(creds)


def _save_credentials(creds: Credentials) -> None:
    os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
        json.dump(
            {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            },
            f,
        )
    # Local-only secret material — do not commit. See .gitignore.
    os.chmod(TOKEN_PATH, 0o600)


def _load_credentials() -> Credentials | None:
    if not os.path.exists(TOKEN_PATH):
        return None
    with open(TOKEN_PATH) as f:
        data = json.load(f)

    creds = Credentials(
        token=data.get("token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri"),
        client_id=data.get("client_id"),
        client_secret=data.get("client_secret"),
        scopes=data.get("scopes"),
    )

    if creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest())
        _save_credentials(creds)

    return creds


def is_connected() -> bool:
    """Cheap check for the frontend to decide whether to show
    'Connect Gmail' or 'Gmail connected' + enable the Send button."""
    creds = _load_credentials()
    return creds is not None and creds.valid


def disconnect() -> None:
    """Local disconnect: deletes the stored token. Does not revoke
    the grant on Google's side — mention that to the user if they
    want a full revoke (they'd do it from myaccount.google.com)."""
    if os.path.exists(TOKEN_PATH):
        os.remove(TOKEN_PATH)


def send_email(
    to: str,
    subject: str,
    body_text: str,
    attachments: list[tuple[str, bytes, str]] | None = None,
) -> dict:
    """
    Send an email via Gmail API. Called ONLY after explicit user
    confirmation in the review-gate UI — see module docstring.

    attachments: list of (filename, file_bytes, mime_type) tuples.
    Returns: {"id": gmail_message_id, "threadId": ...}
    Raises: RuntimeError if not connected, HttpError on API failure.
    """
    creds = _load_credentials()
    if creds is None or not creds.valid:
        raise RuntimeError("Gmail is not connected. Call /api/gmail/auth-url first.")

    message = MIMEMultipart()
    message["to"] = to
    message["subject"] = subject
    message.attach(MIMEText(body_text, "plain"))

    for filename, file_bytes, mime_type in attachments or []:
        maintype, subtype = (mime_type.split("/", 1) if "/" in mime_type else ("application", "octet-stream"))
        part = MIMEApplication(file_bytes, _subtype=subtype)
        part.add_header("Content-Disposition", "attachment", filename=filename)
        message.attach(part)

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    service = build("gmail", "v1", credentials=creds)
    try:
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    except HttpError as e:
        # Surface a clean message rather than the raw googleapiclient traceback
        raise RuntimeError(f"Gmail API send failed: {e}") from e

    return {"id": sent.get("id"), "threadId": sent.get("threadId")}