"""Kaggle credentials for the dataset downloads.

Two kinds are understood, tried in this order:

  1. the newer API token ("KGAT_..."), from the KAGGLE_API_TOKEN environment variable or the file
     ~/.kaggle/access_token, sent as `Authorization: Bearer <token>`
  2. the legacy username + key pair, from KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json,
     sent as HTTP Basic auth

The secret is handed to curl through a config read from STDIN (`curl -K -`), so it never appears in
the process list (`ps`) or in any log line. Nothing here ever prints a credential.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _clean(value: str) -> str:
    v = value.strip()
    if any(ord(c) < 33 or ord(c) == 127 for c in v):
        raise ValueError("a Kaggle credential contains whitespace or control characters")
    return v


def resolve() -> dict | None:
    """-> {"kind": "bearer", "token": ...} | {"kind": "basic", "user": ..., "key": ...} | None."""
    home = Path.home() / ".kaggle"
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token and (home / "access_token").exists():
        token = (home / "access_token").read_text().strip()
    if token:
        return {"kind": "bearer", "token": _clean(token)}
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return {"kind": "basic", "user": _clean(user), "key": _clean(key)}
    if (home / "kaggle.json").exists():
        d = json.loads((home / "kaggle.json").read_text())
        return {"kind": "basic", "user": _clean(d["username"]), "key": _clean(d["key"])}
    return None


def available() -> bool:
    try:
        return resolve() is not None
    except (ValueError, KeyError, OSError):
        return False


def require() -> dict:
    try:
        creds = resolve()
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f"Kaggle credentials are present but unusable: {e}")
    if creds is None:
        sys.exit("No Kaggle credentials. Set KAGGLE_API_TOKEN (new-style token) or "
                 "KAGGLE_USERNAME + KAGGLE_KEY, or create ~/.kaggle/access_token or ~/.kaggle/kaggle.json "
                 "(Kaggle > Settings > API).")
    return creds


def describe(creds: dict) -> str:
    """Which kind is in use - safe to log."""
    return "API token (Bearer)" if creds["kind"] == "bearer" else "username + key (Basic)"


def curl_config(creds: dict) -> str:
    """The text to feed to `curl -K -` on stdin."""
    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if creds["kind"] == "bearer":
        return "header = " + q("Authorization: Bearer " + creds["token"]) + "\n"
    return "user = " + q(creds["user"] + ":" + creds["key"]) + "\n"
