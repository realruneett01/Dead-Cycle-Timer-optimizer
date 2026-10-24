"""Hardware profiles chosen automatically from the probed host."""
from dataclasses import dataclass

from deploy.hostprobe import HostInfo

ARM_32BIT = ("armv6l", "armv7l", "arm")


@dataclass(frozen=True)
class Profile:
    """Resource-appropriate service set and cadence."""
    name: str
    dashboard: bool
    simulator_speedup: float
    checkpoint_every_cycles: int


EDGE = Profile("edge", dashboard=False, simulator_speedup=20.0, checkpoint_every_cycles=25)
STANDARD = Profile("standard", dashboard=True, simulator_speedup=20.0, checkpoint_every_cycles=50)
SERVER = Profile("server", dashboard=True, simulator_speedup=50.0, checkpoint_every_cycles=100)


def select_profile(host: HostInfo) -> Profile:
    """Edge boxes (<=2 CPU, <2 GB, 32-bit ARM) run headless; big hosts get the full stack.

    Unknown memory (0) is treated as unconstrained rather than forcing edge mode.
    """
    low_mem = 0 < host.memory_mb < 2048
    if host.cpu_count <= 2 or low_mem or host.arch in ARM_32BIT:
        return EDGE
    if host.cpu_count >= 8 and host.memory_mb >= 8192:
        return SERVER
    return STANDARD
