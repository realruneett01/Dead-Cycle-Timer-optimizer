"""Detects the capabilities of the machine DCTO is being installed on."""
import importlib.util
import os
import platform
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REQUIRED_MODULES = ["asyncua", "numpy", "scipy", "ruptures", "hmmlearn", "pandas"]
OPTIONAL_MODULES = ["streamlit", "plotly", "matplotlib"]
CGROUP_UNLIMITED = 1 << 60


@dataclass
class HostInfo:
    """Snapshot of host capabilities relevant to profile selection."""
    # pylint: disable=too-many-instance-attributes
    os_name: str
    arch: str
    cpu_count: int
    memory_mb: int
    python_version: str
    in_container: bool
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    numpy_compatible: bool = True

    def to_dict(self) -> Dict:
        """Returns a JSON-serializable view."""
        return asdict(self)


def _cpu_count() -> int:
    """Counts CPUs this process may use (honours affinity and container pinning)."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _read_int(path: str) -> Optional[int]:
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
        return int(raw) if raw.isdigit() else None
    except OSError:
        return None


def _meminfo_mb() -> Optional[int]:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


def _sysctl_mb() -> Optional[int]:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True, timeout=5
        )
        return int(out.stdout.strip()) // (1024 * 1024)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _windows_mb() -> Optional[int]:
    try:
        import ctypes  # pylint: disable=import-outside-toplevel

        class MemStatus(ctypes.Structure):
            """MEMORYSTATUSEX."""
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MemStatus()
        stat.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return int(stat.ullTotalPhys) // (1024 * 1024)
    except (AttributeError, OSError, ImportError):
        return None


def _memory_mb() -> int:
    """Total memory available to this process, honouring cgroup limits."""
    host = _meminfo_mb() or _sysctl_mb() or _windows_mb() or 0
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        limit = _read_int(path)
        if limit and limit < CGROUP_UNLIMITED:
            limit_mb = limit // (1024 * 1024)
            host = min(host, limit_mb) if host else limit_mb
    return host


def _in_container() -> bool:
    if Path("/.dockerenv").exists() or os.environ.get("DCTO_IN_CONTAINER") == "1":
        return True
    try:
        text = Path("/proc/1/cgroup").read_text(encoding="utf-8")
        return any(tok in text for tok in ("docker", "kubepods", "containerd"))
    except OSError:
        return False


def _missing(modules: List[str]) -> List[str]:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def _numpy_compatible() -> bool:
    """requirements.txt pins numpy<2, which only ships wheels up to Python 3.12."""
    return sys.version_info < (3, 13)


def probe_host() -> HostInfo:
    """Collects a HostInfo snapshot for the current machine."""
    return HostInfo(
        os_name=platform.system(),
        arch=platform.machine().lower(),
        cpu_count=_cpu_count(),
        memory_mb=_memory_mb(),
        python_version=platform.python_version(),
        in_container=_in_container(),
        missing_required=_missing(REQUIRED_MODULES),
        missing_optional=_missing(OPTIONAL_MODULES),
        numpy_compatible=_numpy_compatible(),
    )


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """True when nothing is listening on the TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) != 0


def first_free_port(preferred: int, span: int = 50) -> int:
    """Returns the preferred port, or the next free one within span."""
    for port in range(preferred, preferred + span):
        if is_port_free(port):
            return port
    raise RuntimeError(f"No free TCP port in {preferred}-{preferred + span - 1}")


def is_dir_writable(path: Path) -> bool:
    """True when files can be created under path (creating it if needed)."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path):
            return True
    except OSError:
        return False
