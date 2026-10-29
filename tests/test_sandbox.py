"""
URL inspection sandbox.

Fetching an attacker-supplied URL is an SSRF primitive, so most of what
matters here is what the inspector refuses to do.
"""

import pytest

from scamshield.config import Settings
from scamshield.sandbox.guard import BlockedTarget, is_public_address, resolve_and_check
from scamshield.sandbox.inspector import SandboxDisabled, inspect_url


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1",
    "169.254.169.254",      # cloud metadata - leaks instance credentials
    "100.64.0.1", "0.0.0.0", "::1", "fe80::1", "fc00::1",
])
def test_private_addresses_are_not_public(ip):
    assert not is_public_address(ip)


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34"])
def test_public_addresses_allowed(ip):
    assert is_public_address(ip)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "gopher://x/", "ftp://example.com/",
    "javascript:alert(1)", "data:text/html,x",
])
def test_non_http_schemes_blocked(url):
    with pytest.raises(BlockedTarget):
        resolve_and_check(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/", "http://localhost/",
                                 "http://169.254.169.254/latest/meta-data/"])
def test_internal_targets_blocked(url):
    with pytest.raises(BlockedTarget):
        resolve_and_check(url)


def test_url_without_host_blocked():
    with pytest.raises(BlockedTarget):
        resolve_and_check("http://")


@pytest.mark.anyio
async def test_inspection_is_off_by_default():
    """Fetching reveals the caller's IP, so it must never be implicit."""
    with pytest.raises(SandboxDisabled):
        await inspect_url("https://example.com", Settings())


@pytest.mark.anyio
async def test_enabled_sandbox_still_refuses_internal_targets():
    cfg = Settings(sandbox_enabled=True)
    result = await inspect_url("http://169.254.169.254/latest/meta-data/", cfg)
    assert not result.ok
    assert "blocked" in result.error
    assert result.risk_score >= 0.85
