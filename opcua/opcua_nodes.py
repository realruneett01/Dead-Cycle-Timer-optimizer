"""ISA-95 industrial automation node definitions for OPC-UA Server."""
from dataclasses import dataclass
from typing import Any, Dict

NAMESPACE_URI = "urn:industrial:press:telemetry"

@dataclass
class PressNodeRefs:
    """Holds references to instantiated asyncua Node objects."""
    # State Nodes
    current_phase: Any = None
    cycle_number: Any = None
    phase_start_time: Any = None
    phase_duration: Any = None
    is_dead_cycle: Any = None
    
    # Telemetry Nodes
    main_cylinder_pressure: Any = None
    proportional_valve_spool: Any = None
    container_position: Any = None
    emergency_stop: Any = None
    
    # Diagnostics Nodes
    active_stall_flag: Any = None
    estimated_excess_delay: Any = None

async def build_press_node_hierarchy(server, ns_idx: int) -> PressNodeRefs:
    """
    Constructs the ISA-95 hierarchical node structure under Root/Objects/Industrial_Plant/Press_01/
    and returns a dataclass of Node references for low-latency writes.
    """
    objects = server.nodes.objects
    
    # Corporate / Plant Level Object
    plant_folder = await objects.add_folder(ns_idx, "Industrial_Plant")
    press_01 = await plant_folder.add_object(ns_idx, "Press_01")
    
    # 1. State Sub-folder
    state_folder = await press_01.add_folder(ns_idx, "State")
    current_phase = await state_folder.add_variable(ns_idx, "CurrentPhase", "idle")
    cycle_number = await state_folder.add_variable(ns_idx, "CycleNumber", 0)
    phase_start_time = await state_folder.add_variable(ns_idx, "PhaseStartTime", "")
    phase_duration = await state_folder.add_variable(ns_idx, "PhaseDuration", 0.0)
    is_dead_cycle = await state_folder.add_variable(ns_idx, "IsDeadCycle", True)
    
    # 2. Telemetry Sub-folder
    telem_folder = await press_01.add_folder(ns_idx, "Telemetry")
    main_cylinder_pressure = await telem_folder.add_variable(ns_idx, "MainCylinderPressure", 0.0)
    proportional_valve_spool = await telem_folder.add_variable(ns_idx, "ProportionalValveSpool", 0.0)
    container_position = await telem_folder.add_variable(ns_idx, "ContainerPosition", 0.0)
    emergency_stop = await telem_folder.add_variable(ns_idx, "EmergencyStop", False)
    
    # 3. Diagnostics Sub-folder
    diag_folder = await press_01.add_folder(ns_idx, "Diagnostics")
    active_stall_flag = await diag_folder.add_variable(ns_idx, "ActiveStallFlag", False)
    estimated_excess_delay = await diag_folder.add_variable(ns_idx, "EstimatedExcessDelay", 0.0)
    
    # Make all telemetry and diagnostics writable by authorized clients
    for node in [
        current_phase, cycle_number, phase_start_time, phase_duration, is_dead_cycle,
        main_cylinder_pressure, proportional_valve_spool, container_position, emergency_stop,
        active_stall_flag, estimated_excess_delay
    ]:
        await node.set_writable()

    return PressNodeRefs(
        current_phase=current_phase,
        cycle_number=cycle_number,
        phase_start_time=phase_start_time,
        phase_duration=phase_duration,
        is_dead_cycle=is_dead_cycle,
        main_cylinder_pressure=main_cylinder_pressure,
        proportional_valve_spool=proportional_valve_spool,
        container_position=container_position,
        emergency_stop=emergency_stop,
        active_stall_flag=active_stall_flag,
        estimated_excess_delay=estimated_excess_delay
    )
