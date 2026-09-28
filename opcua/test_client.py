"""Verification test client for subscribing to OPC-UA press telemetry."""
import asyncio
import logging
import sys
from pathlib import Path

from asyncua import Client
import opcua.opcua_nodes as nodes

logger = logging.getLogger("OPCUAClient")

class PressSubscriptionHandler:
    """Handles asynchronous data change notifications from the OPC-UA server."""

    def __init__(self, node_names: dict, max_notifications: int = 20):
        self.node_names = node_names
        self.notification_count = 0
        self.max_notifications = max_notifications
        self.done_event = asyncio.Event()

    def datachange_notification(self, node, val, data):
        # pylint: disable=unused-argument
        """Callback invoked by asyncua on node value change."""
        self.notification_count += 1
        name = self.node_names.get(str(node), str(node))
        logger.info(
            "[TELEMETRY UPDATE #%03d] Node: %-22s => Value: %s",
            self.notification_count, name, val
        )

        if self.notification_count >= self.max_notifications:
            self.done_event.set()

    def event_notification(self, event):
        # pylint: disable=unused-argument
        """Callback for event notifications."""

async def run_client(
    endpoint: str = nodes.DEFAULT_OPCUA_ENDPOINT,
    max_events: int = 20
):
    # pylint: disable=too-many-locals
    """Connects to server, subscribes to key nodes, and logs incoming events."""
    logger.info("Connecting to OPC-UA server at %s...", endpoint)

    async with Client(url=endpoint) as client:
        # Resolve namespace index
        ns_idx = await client.get_namespace_index(nodes.NAMESPACE_URI)
        logger.info("Resolved namespace '%s' to index %s", nodes.NAMESPACE_URI, ns_idx)

        # Locate target nodes
        objects = client.nodes.objects

        # Navigate to target paths
        phase_path = nodes.get_press_node_path(ns_idx, "State", "CurrentPhase")
        cycle_path = nodes.get_press_node_path(ns_idx, "State", "CycleNumber")
        dur_path = nodes.get_press_node_path(ns_idx, "State", "PhaseDuration")
        stall_path = nodes.get_press_node_path(ns_idx, "Diagnostics", "ActiveStallFlag")

        phase_node = await objects.get_child(phase_path)
        cycle_node = await objects.get_child(cycle_path)
        duration_node = await objects.get_child(dur_path)
        stall_node = await objects.get_child(stall_path)

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
        logger.info(
            "Subscribed to 4 nodes. Waiting for %d telemetry updates...", max_events
        )

        try:
            await asyncio.wait_for(handler.done_event.wait(), timeout=30.0)
            logger.info("Successfully received target notification count from OPC-UA server.")
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for subscription notifications.")

        await sub.delete()

def main():
    """CLI driver for running the verification test client."""
    parser = nodes.create_port_parser("Test client for OPC-UA Server.")
    parser.add_argument(
        "--events", type=int, default=20, help="Number of telemetry events to capture"
    )
    args = parser.parse_args()
    endpoint = nodes.build_opcua_endpoint(args.port)
    asyncio.run(run_client(endpoint=endpoint, max_events=args.events))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Client: %(message)s")
    main()
