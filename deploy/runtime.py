"""Production runtime: OPC-UA telemetry (real gateway or built-in simulator) -> DetectorService."""
import asyncio
import logging
from typing import Dict, List, Optional

from asyncua import Client, Node, ua

from deploy.press_profile import load_press_config
from deploy.settings import Settings
from detector.detector_service import DetectorService, DetectorServiceConfig, PhaseEventObservation
from opcua.opcua_nodes import NAMESPACE_URI

logger = logging.getLogger("DCTORuntime")

LOGICAL_NODES = {
    "phase": "CurrentPhase",
    "cycle": "CycleNumber",
    "duration": "PhaseDuration",
}
MAX_BROWSE_DEPTH = 6
SAMPLING_MS = 20  # must be well below the shortest phase duration


async def _find_by_browse_name(root: Node, name: str, depth: int = MAX_BROWSE_DEPTH) -> Optional[Node]:
    """Breadth-first search for a node by browse name, independent of plant/press folder layout."""
    level: List[Node] = [root]
    for _ in range(depth):
        nxt: List[Node] = []
        for node in level:
            for child in await node.get_children():
                if (await child.read_browse_name()).Name == name:
                    return child
                nxt.append(child)
        level = nxt
    return None


async def discover_nodes(client: Client, overrides: Dict[str, str]) -> Dict[str, Node]:
    """Locates phase/cycle/duration nodes by browse name; names are overridable per plant."""
    objects = client.nodes.objects
    found: Dict[str, Node] = {}
    for logical, default in LOGICAL_NODES.items():
        wanted = overrides.get(logical, default)
        node = await _find_by_browse_name(objects, wanted)
        if node is None:
            raise LookupError(
                f"OPC-UA node '{wanted}' not found. Set node_names.{logical} in config/dcto.json "
                f"(or DCTO_NODE_NAMES) to this plant's browse name."
            )
        found[logical] = node
    return found


class TelemetryTap:
    """Subscribes to phase/cycle/duration and emits (cycle, phase, duration) snapshots.

    Phase and cycle are taken from the subscription stream, not re-read on each duration
    update: a fresh read can race the publisher and attribute a duration to the wrong phase.
    """

    def __init__(self, nodes: Dict[str, Node]):
        self._ids = {nodes[k].nodeid: k for k in LOGICAL_NODES}
        self._nodes = nodes
        self._latest: Dict[str, object] = {}
        self._phase_updates = 0
        self.dropped = 0
        self.queue: asyncio.Queue = asyncio.Queue()

    def datachange_notification(self, node, val, data):
        # pylint: disable=unused-argument
        """asyncua callback; duration updates complete a snapshot."""
        key = self._ids.get(node.nodeid)
        if key is None:
            return
        self._latest[key] = val
        if key == "phase":
            self._phase_updates += 1
        if key == "duration" and "phase" in self._latest and "cycle" in self._latest:
            # More than one phase change since the last duration means updates were skipped, so
            # the pairing is unreliable: drop it rather than mislabel it.
            if self._phase_updates > 1:
                self.dropped += 1
            else:
                self.queue.put_nowait(
                    (int(self._latest["cycle"]), str(self._latest["phase"]), float(val)))
            self._phase_updates = 0

    async def start(self, client: Client):
        """Creates the subscription; returns it so the caller can delete it."""
        sub = await client.create_subscription(SAMPLING_MS, self)
        # Initial values arrive for phase/cycle first, then duration, in this order.
        await sub.subscribe_data_change([self._nodes["phase"], self._nodes["cycle"], self._nodes["duration"]])
        return sub


class _Ingest:
    """Feeds tap snapshots to the detector."""

    def __init__(self, service: DetectorService, tap: TelemetryTap, settings: Settings):
        self.service = service
        self.tap = tap
        self.settings = settings
        self.processed = 0

    async def drain(self, stop: asyncio.Event):
        """Processes queued snapshots until stop is set."""
        while not stop.is_set():
            try:
                cycle, phase, duration = await asyncio.wait_for(self.tap.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            record = self.service.process_phase_event(PhaseEventObservation(cycle, phase, duration))
            self.processed += 1
            if self.tap.dropped and self.processed % 100 == 0:
                logger.warning("%d telemetry updates skipped (source faster than %d ms sampling)",
                               self.tap.dropped, SAMPLING_MS)
            if record["is_anomaly"]:
                logger.warning(
                    "ANOMALY cycle=%s phase=%s type=%s excess=%.2fs",
                    cycle, phase, record["anomaly_type"], record["excess_seconds"],
                )
            if self.processed % self.settings.checkpoint_every_cycles == 0:
                self.service.save_model_checkpoint(str(self.settings.checkpoint_path))


def build_service(settings: Settings) -> DetectorService:
    """Creates the detector for this machine and restores its checkpoint if one exists."""
    service = DetectorService(DetectorServiceConfig(
        config=load_press_config(settings.press_profile or None),
        telemetry_log_path=str(settings.telemetry_path),
        threshold_z=settings.threshold_z,
        cusum_alpha=settings.cusum_alpha,
        cusum_beta=settings.cusum_beta,
        append_to_log=True,
    ))
    service.load_model_checkpoint(str(settings.checkpoint_path))
    return service


async def _session(settings: Settings, endpoint: str, service: DetectorService, stop: asyncio.Event):
    async with Client(url=endpoint) as client:
        try:
            await client.get_namespace_index(NAMESPACE_URI)
        except (ValueError, ua.UaError):
            logger.info("Namespace %s absent; relying on browse-name discovery", NAMESPACE_URI)
        nodes = await discover_nodes(client, settings.node_names)
        tap = TelemetryTap(nodes)
        sub = await tap.start(client)
        logger.info("Streaming %s -> detector (profile=%s)", endpoint, settings.profile)
        await _Ingest(service, tap, settings).drain(stop)
        await sub.delete()


async def run_ingest(settings: Settings, endpoint: str, stop: asyncio.Event):
    """Connects with exponential-backoff reconnect; detector state survives reconnects."""
    service = build_service(settings)
    delay = 1.0
    try:
        while not stop.is_set():
            try:
                await _session(settings, endpoint, service, stop)
                delay = 1.0
            except (OSError, asyncio.TimeoutError, ua.UaError, ConnectionError) as exc:
                logger.warning("OPC-UA link down (%s); retrying in %.0fs", exc, delay)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(delay * 2, settings.reconnect_max_sec)
    finally:
        service.save_model_checkpoint(str(settings.checkpoint_path))
        service.close()
