"""Finds OPC-UA servers (PLC gateways / sensor hubs) reachable from this machine."""
import asyncio
import ipaddress
import socket
from typing import Iterable, List, Optional

from asyncua import Client

DEFAULT_PORTS = tuple(range(4840, 4851))
CONNECT_TIMEOUT = 0.4


def local_subnet() -> Optional[str]:
    """The /24 of this host's primary IPv4 address (no packets are sent by the UDP trick)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))
            ip = sock.getsockname()[0]
        return str(ipaddress.ip_network(f"{ip}/24", strict=False))
    except OSError:
        return None


async def _tcp_open(host: str, port: int, sem: asyncio.Semaphore) -> bool:
    async with sem:
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), CONNECT_TIMEOUT)
        except (OSError, asyncio.TimeoutError):
            return False
        writer.close()
        return True


async def _is_opcua(url: str) -> bool:
    try:
        async with Client(url=url, timeout=3):
            return True
    except Exception:  # pylint: disable=broad-except
        return False


async def find_endpoints(
    hosts: Iterable[str] = ("127.0.0.1",),
    ports: Iterable[int] = DEFAULT_PORTS,
    path: str = "",
) -> List[str]:
    """TCP-probes host/port pairs, then confirms the OPC-UA handshake on the open ones."""
    sem = asyncio.Semaphore(256)
    pairs = [(h, p) for h in hosts for p in ports]
    opened = await asyncio.gather(*(_tcp_open(h, p, sem) for h, p in pairs))
    urls = [f"opc.tcp://{h}:{p}{path}" for (h, p), ok in zip(pairs, opened) if ok]
    verified = await asyncio.gather(*(_is_opcua(u) for u in urls))
    return [u for u, ok in zip(urls, verified) if ok]


def subnet_hosts(cidr: str) -> List[str]:
    """Usable host addresses of a CIDR block."""
    return [str(h) for h in ipaddress.ip_network(cidr, strict=False).hosts()]
