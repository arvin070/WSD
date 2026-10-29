"""
Network guards for the URL inspector.

Fetching an attacker-supplied URL from your own infrastructure is a
server-side request forgery primitive unless it is fenced in. The fence has
three parts, and all three matter:

1. **Scheme allowlist.** Only http and https. `file:`, `gopher:` and friends
   reach places HTTP cannot.
2. **Address filtering.** Every address a hostname resolves to is checked, not
   just the first. A host with one public and one loopback record would
   otherwise slip through, and `169.254.169.254` is the cloud metadata
   endpoint that leaks instance credentials.
3. **Per-hop revalidation.** A public URL can redirect to `127.0.0.1`, so the
   check runs again on every hop rather than once at the start.

**Known limitation:** the resolve-then-connect gap allows DNS rebinding — a
name that answers with a public address for our check and a private one for
the connection. Closing it properly needs connection-level IP pinning. The
exposure is bounded because nothing fetched is ever executed, written to
disk, or echoed to another internal service.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

ALLOWED_SCHEMES = {"http", "https"}
#: Ranges that must never be reachable from a URL a stranger sent.
BLOCKED_EXTRA = [
    ipaddress.ip_network("169.254.0.0/16"),   # link-local incl. cloud metadata
    ipaddress.ip_network("100.64.0.0/10"),    # carrier-grade NAT
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("198.18.0.0/15"),    # benchmarking
    ipaddress.ip_network("::ffff:0:0/96"),    # IPv4-mapped IPv6
    ipaddress.ip_network("fc00::/7"),         # unique local
    ipaddress.ip_network("fe80::/10"),        # link-local v6
]


class BlockedTarget(RuntimeError):
    """The URL points somewhere the inspector refuses to go."""


@dataclass(frozen=True)
class ResolvedTarget:
    host: str
    port: int
    addresses: tuple[str, ...]


def is_public_address(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if (addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_multicast or addr.is_reserved or addr.is_unspecified):
        return False
    return all(addr not in net for net in BLOCKED_EXTRA
               if addr.version == net.version)


def resolve_and_check(url: str, allow_private: bool = False) -> ResolvedTarget:
    """
    Validate a URL and resolve it, refusing anything non-public.

    `allow_private` exists so the test suite can point at a local server; it
    must never be enabled in a deployment that accepts untrusted URLs.
    """
    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise BlockedTarget(f"scheme '{scheme or 'none'}' is not allowed")
    host = parts.hostname
    if not host:
        raise BlockedTarget("URL has no host")
    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise BlockedTarget("malformed port") from exc

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BlockedTarget(f"host does not resolve: {host}") from exc

    addresses = tuple(dict.fromkeys(info[4][0] for info in infos))
    if not addresses:
        raise BlockedTarget(f"host does not resolve: {host}")
    if not allow_private:
        # Every address, not just the first: one private record is enough.
        bad = [a for a in addresses if not is_public_address(a)]
        if bad:
            raise BlockedTarget(f"{host} resolves to a non-public address ({bad[0]})")
    return ResolvedTarget(host=host, port=port, addresses=addresses)
