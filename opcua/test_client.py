"""Verification test client for subscribing to OPC-UA press telemetry."""
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asyncua import Client
from opcua.opcua_nodes import NAMESPACE_URI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Client: %(message)s")
logger = logging.getLogger("OPCUAClient")

class PressSubscriptionHandler:
    """Handles asynchronous data change notifications from the OPC-UA server."""

    def __init__(self, node_names: dict, max_notifications: int = 20):
        self.node_names = node_names
        self.notification_count = 0
        self.max_notifications = max_notifications
        self.done_event = asyncio.Event()

    def datachange_notification(self, node, val, data):
        self.notification_count += 1
        name = self.node_names.get(str(node), str(node))
        logger.info(f"[TELEMETRY UPDATE #{self.notification_count:03d}] Node: {name:<22} => Value: {val}")
        
        if self.notification_count >= self.max_notifications:
            self.done_event.set()

    def event_notification(self, event):
        pass

async def run_client(
    endpoint: str = "opc.tcp://127.0.0.1:4840/freeopcua/server/",
    max_events: int = 20
):
    """Connects to server, subscribes to key nodes, and logs incoming events."""
    logger.info(f"Connecting to OPC-UA server at {endpoint}...")
    
    async with Client(url=endpoint) as client:
        # Resolve namespace index
        ns_idx = await client.get_namespace_index(NAMESPACE_URI)
        logger.info(f"Resolved namespace '{NAMESPACE_URI}' to index {ns_idx}")

        # Locate target nodes
        root = client.nodes.root
        objects = client.nodes.objects
        
        # Navigate to Industrial_Plant/Press_01/State
        phase_node = await objects.get_child([f"{ns_idx}:Industrial_Plant", f"{ns_idx}:Press_01", f"{ns_idx}:State", f"{ns_idx}:CurrentPhase"])
        cycle_node = await objects.get_child([f"{ns_idx}:Industrial_Plant", f"{ns_idx}:Press_01", f"{ns_idx}:State", f"{ns_idx}:CycleNumber"])
        duration_node = await objects.get_child([f"{ns_idx}:Industrial_Plant", f"{ns_idx}:Press_01", f"{ns_idx}:State", f"{ns_idx}:PhaseDuration"])
        stall_node = await objects.get_child([f"{ns_idx}:Industrial_Plant", f"{ns_idx}:Press_01", f"{ns_idx}:Diagnostics", f"{ns_idx}:ActiveStallFlag"])

        node_map = {
            str(phase_node): "CurrentPhase",
            str(cycle_node): "CycleNumber",
            str(duration_node): "PhaseDuration",
            str(stall_node): "ActiveStallFlag",
        }

        # Create subscription
        handler = PressSubscriptionHandler(node_names=node_map, max_notifications=max_events)
        sub = await client.create_subscription(50, handler)
        
        await sub.subscribe_data_change([phase_node, cycle_node, duration_node, stall_node])
        logger.info(f"Subscribed to 4 nodes. Waiting for {max_events} telemetry updates...")

        try:
            await asyncio.wait_for(handler.done_event.wait(), timeout=30.0)
            logger.info("Successfully received target notification count from OPC-UA server.")
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for subscription notifications.")

        await sub.delete()

def main():
    parser = argparse.ArgumentParser(description="Test client for OPC-UA Server.")
    parser.add_argument("--port", type=int, default=4840, help="Port of OPC-UA server")
    parser.add_argument("--events", type=int, default=20, help="Number of telemetry events to capture")
    args = parser.parse_args()

    endpoint = f"opc.tcp://127.0.0.1:{args.port}/freeopcua/server/"
    asyncio.run(run_client(endpoint=endpoint, max_events=args.events))

if __name__ == "__main__":
    main()
