"""Deployment settings: JSON file + DCTO_* environment overrides + host-derived defaults."""
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

from deploy.hostprobe import HostInfo, first_free_port, probe_host
from deploy.profiles import select_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "dcto.json"
ENV_PREFIX = "DCTO_"


@dataclass
class Settings:
    """Everything the runtime needs; every field has a host-derived or safe default."""
    # pylint: disable=too-many-instance-attributes
    profile: str = "standard"
    endpoint: str = ""                  # empty => run the built-in simulator locally
    simulate: Optional[bool] = None     # None => simulate only when no endpoint is set
    simulator_port: int = 4840
    simulator_speedup: float = 20.0
    node_names: Dict[str, str] = field(default_factory=dict)  # logical -> browse-name override
    dashboard: bool = True
    dashboard_port: int = 8501
    data_dir: str = str(PROJECT_ROOT / "data")
    press_profile: str = ""             # path to press profile JSON (see press_profile.py)
    checkpoint_every_cycles: int = 50
    threshold_z: float = 2.75
    cusum_alpha: float = 0.01
    cusum_beta: float = 0.05
    reconnect_max_sec: float = 30.0

    @property
    def use_simulator(self) -> bool:
        """Whether to start the built-in OPC-UA simulator instead of a real PLC gateway."""
        return self.simulate if self.simulate is not None else not self.endpoint

    @property
    def telemetry_path(self) -> Path:
        """Telemetry CSV the dashboard's 'Load CSV Stream' mode reads."""
        return Path(self.data_dir) / "telemetry_stream.csv"

    @property
    def checkpoint_path(self) -> Path:
        """Detector checkpoint, restored on restart so baselines survive reboots."""
        return Path(self.data_dir) / "detector_checkpoint.json"


def _coerce(raw: str, current: Any) -> Any:
    if current is None or isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, dict):
        return json.loads(raw)
    return raw


def derive_defaults(host: HostInfo) -> Settings:
    """Builds Settings tuned to the host: profile, free ports, service set."""
    profile = select_profile(host)
    return Settings(
        profile=profile.name,
        dashboard=profile.dashboard and "streamlit" not in host.missing_optional,
        simulator_speedup=profile.simulator_speedup,
        checkpoint_every_cycles=profile.checkpoint_every_cycles,
        simulator_port=first_free_port(4840),
        dashboard_port=first_free_port(8501),
    )


def load_settings(path: Optional[Path] = None, host: Optional[HostInfo] = None) -> Settings:
    """Merges host-derived defaults < settings file < DCTO_* env vars."""
    settings = derive_defaults(host or probe_host())
    src = Path(path) if path else DEFAULT_SETTINGS_PATH
    if src.exists():
        data = json.loads(src.read_text(encoding="utf-8"))
        known = {f.name for f in fields(Settings)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown setting(s) in {src}: {', '.join(unknown)}")
        for key, value in data.items():
            setattr(settings, key, value)
    for f in fields(Settings):
        raw = os.environ.get(ENV_PREFIX + f.name.upper())
        if raw is not None:
            setattr(settings, f.name, _coerce(raw, getattr(settings, f.name)))
    return settings


def save_settings(settings: Settings, path: Optional[Path] = None) -> Path:
    """Writes settings JSON (creating the config directory)."""
    dest = Path(path) if path else DEFAULT_SETTINGS_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")
    return dest
