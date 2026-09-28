"""Asynchronous IEC 62541 OPC-UA server for Extrusion Press."""
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=wrong-import-position
from asyncua import Server
from opcua.opcua_nodes import (
    DEFAULT_OPCUA_ENDPOINT,
    NAMESPACE_URI,
    PressNodeRefs,
    build_opcua_endpoint,
    build_press_node_hierarchy,
    create_port_parser,
)
from simulator.press_state_machine import CycleEvent, PressStateMachine

logger = logging.getLogger("OPCUAServer")

class PressOpcUaServer:
    """Manages lifecycle of the OPC-UA server and streams press state machine telemetry."""

    def __init__(
        self,
        endpoint: str = DEFAULT_OPCUA_ENDPOINT,
        server_name: str = "Extrusion Press 01 OPC-UA Server",
        state_machine: Optional[PressStateMachine] = None
    ):
        self.endpoint = endpoint
        self.server_name = server_name
        self.state_machine = state_machine or PressStateMachine()
        self.server = Server()
        self.nodes: Optional[PressNodeRefs] = None
        self.ns_idx: int = 0
        self._is_running = False

    async def init(self):
        """Initializes server configuration and ISA-95 node space."""
        await self.server.init()
        self.server.set_endpoint(self.endpoint)
        self.server.set_server_name(self.server_name)

        # Register namespace
        self.ns_idx = await self.server.register_namespace(NAMESPACE_URI)
        self.nodes = await build_press_node_hierarchy(self.server, self.ns_idx)
        logger.info(
            "Initialized %s with namespace '%s' (index=%s)",
            self.server_name, NAMESPACE_URI, self.ns_idx
        )

    async def publish_event(self, event: CycleEvent):
        """Updates OPC-UA node values with a new cycle event."""
        if not self.nodes:
            return

        await self.nodes.current_phase.write_value(event.phase_name)
        await self.nodes.cycle_number.write_value(int(event.cycle_id))
        await self.nodes.phase_start_time.write_value(event.start_time)
        await self.nodes.phase_duration.write_value(float(event.duration_sec))
        await self.nodes.is_dead_cycle.write_value(bool(event.is_dead_cycle))
        await self.nodes.main_cylinder_pressure.write_value(float(event.pressure_bar))
        await self.nodes.proportional_valve_spool.write_value(float(event.valve_spool_pct))
        await self.nodes.container_position.write_value(float(event.ram_position_mm))
        await self.nodes.active_stall_flag.write_value(bool(event.has_anomaly))
        await self.nodes.estimated_excess_delay.write_value(float(event.injected_delay_sec))

    async def run(self, max_cycles: Optional[int] = None, speedup: float = 10.0):
        """Starts server and loops through press cycles until stopped."""
        await self.init()
        self._is_running = True

        async with self.server:
            logger.info("OPC-UA server listening on %s", self.endpoint)
            cycle_id = 0

            while self._is_running:
                cycle_id += 1
                if max_cycles is not None and cycle_id > max_cycles:
                    break

                events = self.state_machine.run_cycle(cycle_id=cycle_id)
                for event in events:
                    if not self._is_running:
                        break

                    await self.publish_event(event)
                    logger.debug(
                        "Cycle %s | Phase: %-22s | Duration: %.2fs | Delay: +%.2fs",
                        event.cycle_id, event.phase_name,
                        event.duration_sec, event.injected_delay_sec
                    )

                    # Sleep scaled by speedup factor (e.g. 10x faster than real-time)
                    if speedup > 0:
                        sleep_time = event.duration_sec / speedup
                        await asyncio.sleep(sleep_time)

    def stop(self):
        """Signals server loop to stop."""
        self._is_running = False

async def main():
    """CLI driver for starting and managing the OPC-UA server lifecycle."""
    parser = create_port_parser("Run Extrusion Press OPC-UA Server.")
    parser.add_argument(
        "--speedup", type=float, default=20.0,
        help="Speedup factor relative to real-time (default: 20x)"
    )
    parser.add_argument(
        "--cycles", type=int, default=None, help="Max cycles to run (default: infinite)"
    )
    args = parser.parse_args()

    endpoint = build_opcua_endpoint(args.port)
    server = PressOpcUaServer(endpoint=endpoint)

    try:
        await server.run(max_cycles=args.cycles, speedup=args.speedup)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Stopping OPC-UA server...")
        server.stop()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    asyncio.run(main())
