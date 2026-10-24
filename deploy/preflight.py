"""Static pre-deployment checks. Each returns PASS/WARN/FAIL with a concrete fix."""
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from deploy.hostprobe import HostInfo, is_dir_writable, is_port_free
from deploy.settings import Settings

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MIN_FREE_MB = 200


@dataclass
class Check:
    """One preflight or commissioning check outcome."""
    name: str
    status: str
    detail: str
    fix: str = ""

    def to_dict(self) -> Dict:
        """JSON view."""
        return asdict(self)


def run_preflight(host: HostInfo, settings: Settings) -> List[Check]:
    """Checks the host and storage; endpoint and signal checks happen during commissioning."""
    checks: List[Check] = []

    def add(name, ok, detail, fix="", level=FAIL):
        checks.append(Check(name, PASS if ok else level, detail, "" if ok else fix))

    add("python", host.numpy_compatible, f"Python {host.python_version}",
        "Use Python 3.11 or 3.12 (numpy<2 has no wheels for 3.13+)")
    add("packages", not host.missing_required,
        "all required packages present" if not host.missing_required
        else f"missing: {', '.join(host.missing_required)}",
        "Re-run the launcher so it installs requirements (needs vendor/wheels or internet)")
    if settings.dashboard:
        add("dashboard-packages", not host.missing_optional,
            "dashboard packages present" if not host.missing_optional
            else f"missing: {', '.join(host.missing_optional)}",
            "Install requirements, or run with --no-dashboard", level=WARN)
        add("dashboard-port", is_port_free(settings.dashboard_port),
            f"port {settings.dashboard_port}", "Choose another via DCTO_DASHBOARD_PORT", level=WARN)

    data_dir = Path(settings.data_dir)
    writable = is_dir_writable(data_dir)
    add("data-dir-writable", writable, str(data_dir),
        "The drive is read-only or write-protected; unlock it or set DCTO_DATA_DIR")
    if writable:
        free_mb = shutil.disk_usage(data_dir).free // (1024 * 1024)
        add("data-dir-space", free_mb >= MIN_FREE_MB, f"{free_mb} MB free",
            f"Free at least {MIN_FREE_MB} MB for telemetry and checkpoints")

    add("memory", host.memory_mb == 0 or host.memory_mb >= 512,
        f"{host.memory_mb} MB", "Detector needs ~100 MB; free memory or use a larger host")
    return checks
