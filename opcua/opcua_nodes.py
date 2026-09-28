import argparse
import re
from dataclasses import dataclass
from typing import Any

NAMESPACE_URI = "urn:industrial:press:telemetry"

DEFAULT_OPCUA_ENDPOINT = "opc.tcp://127.0.0.1:4840/freeopcua/server/"


def build_opcua_endpoint(port: int = 4840) -> str:
    """Formats standardized local OPC-UA server endpoint URI."""
    return f"opc.tcp://127.0.0.1:{port}/freeopcua/server/"


def create_port_parser(description: str) -> argparse.ArgumentParser:
    """Creates standard CLI argument parser with port configuration."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--port", type=int, default=4840, help="OPC-UA server port")
    return parser


def _to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class PressNodeRefs:
    """Holds references to instantiated asyncua Node objects."""

    def __init__(self, **nodes: Any):
        for name, node in nodes.items():
            setattr(self, name, node)


FOLDER_ITEMS = {
    "State": [("CurrentPhase", "idle"), ("CycleNumber", 0), ("PhaseStartTime", ""), ("PhaseDuration", 0.0), ("IsDeadCycle", True)],
    "Telemetry": [("MainCylinderPressure", 0.0), ("ProportionalValveSpool", 0.0), ("ContainerPosition", 0.0), ("EmergencyStop", False)],
    "Diagnostics": [("ActiveStallFlag", False), ("EstimatedExcessDelay", 0.0)],
}


async def build_press_node_hierarchy(server, ns_idx: int) -> PressNodeRefs:
    """
    Constructs the ISA-95 hierarchical node structure under Root/Objects/Industrial_Plant/Press_01/
    and returns a dataclass of Node references for low-latency writes.
    """
    objects = server.nodes.objects
    plant_folder = await objects.add_folder(ns_idx, "Industrial_Plant")
    press_01 = await plant_folder.add_object(ns_idx, "Press_01")

    node_kwargs = {}
    for folder_name, variables in FOLDER_ITEMS.items():
        folder = await press_01.add_folder(ns_idx, folder_name)
        for var_name, default_val in variables:
            node = await folder.add_variable(ns_idx, var_name, default_val)
            await node.set_writable()
            node_kwargs[_to_snake(var_name)] = node

    return PressNodeRefs(**node_kwargs)


def get_press_node_path(ns_idx: int, folder: str, variable: str) -> list[str]:
    """Returns the ISA-95 browse path components for a press variable node."""
    return [
        f"{ns_idx}:Industrial_Plant",
        f"{ns_idx}:Press_01",
        f"{ns_idx}:{folder}",
        f"{ns_idx}:{variable}",
    ]
