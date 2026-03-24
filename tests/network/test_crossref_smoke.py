from __future__ import annotations

import os

import pytest


@pytest.mark.network
def test_crossref_email_present_when_running_network() -> None:
    """
    Opt-in smoke: only meaningful when you actually intend to run network tests.
    """
    email = os.environ.get("CROSSREF_EMAIL") or ""
    if not email:
        pytest.skip("CROSSREF_EMAIL not set")
    assert "@" in email

