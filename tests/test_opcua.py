"""Integration tests for OPC-UA server and client communication."""
import asyncio
import pytest
from asyncua import Client
from opcua.opcua_nodes import NAMESPACE_URI
from opcua.opcua_server import PressOpcUaServer
from simulator.press_config import PressConfig
from simulator.press_state_machine import CycleEvent, PressStateMachine

@pytest.mark.asyncio
async def test_opcua_server_initialization_and_read_write():
    test_port = 4842
    endpoint = f"opc.tcp://127.0.0.1:{test_port}/freeopcua/server/"
    
    server = PressOpcUaServer(endpoint=endpoint)
    await server.init()
    
    # Start server in background
    async with server.server:
        # Connect client
        async with Client(url=endpoint) as client:
            ns_idx = await client.get_namespace_index(NAMESPACE_URI)
            assert ns_idx > 0
            
            objects = client.nodes.objects
            phase_node = await objects.get_child([f"{ns_idx}:Industrial_Plant", f"{ns_idx}:Press_01", f"{ns_idx}:State", f"{ns_idx}:CurrentPhase"])
            cycle_node = await objects.get_child([f"{ns_idx}:Industrial_Plant", f"{ns_idx}:Press_01", f"{ns_idx}:State", f"{ns_idx}:CycleNumber"])
            
            # Initial values
            val_phase = await phase_node.read_value()
            assert val_phase == "idle"
            
            # Publish event
            test_event = CycleEvent(
                cycle_id=101,
                phase_name="shear_stroke",
                phase_index=2,
                start_time="2026-09-25T12:00:00Z",
                duration_sec=2.65,
                nominal_duration_sec=2.50,
                is_dead_cycle=True,
                pressure_bar=185.0,
                valve_spool_pct=88.0,
                ram_position_mm=0.0,
                has_anomaly=True,
                anomaly_type="VALVE_OVERLAP",
                injected_delay_sec=0.35
            )
            await server.publish_event(test_event)
            
            # Read back
            updated_phase = await phase_node.read_value()
            updated_cycle = await cycle_node.read_value()
            assert updated_phase == "shear_stroke"
            assert updated_cycle == 101

@pytest.mark.asyncio
async def test_opcua_subscription():
    test_port = 4843
    endpoint = f"opc.tcp://127.0.0.1:{test_port}/freeopcua/server/"
    
    server = PressOpcUaServer(endpoint=endpoint)
    await server.init()
    
    received_updates = []
    
    class SubHandler:
        def datachange_notification(self, node, val, data):
            received_updates.append(val)
    
    async with server.server:
        async with Client(url=endpoint) as client:
            ns_idx = await client.get_namespace_index(NAMESPACE_URI)
            objects = client.nodes.objects
            phase_node = await objects.get_child([f"{ns_idx}:Industrial_Plant", f"{ns_idx}:Press_01", f"{ns_idx}:State", f"{ns_idx}:CurrentPhase"])
            
            handler = SubHandler()
            sub = await client.create_subscription(20, handler)
            await sub.subscribe_data_change(phase_node)
            
            # Give subscription brief moment to register
            await asyncio.sleep(0.1)
            
            # Publish two distinct events
            ev1 = CycleEvent(
                cycle_id=1, phase_name="container_shift_open", phase_index=1,
                start_time="2026-09-25T12:00:00Z", duration_sec=3.0, nominal_duration_sec=3.0,
                is_dead_cycle=True, pressure_bar=50.0, valve_spool_pct=75.0, ram_position_mm=0.0,
                has_anomaly=False, anomaly_type="NONE", injected_delay_sec=0.0
            )
            ev2 = CycleEvent(
                cycle_id=1, phase_name="shear_stroke", phase_index=2,
                start_time="2026-09-25T12:00:03Z", duration_sec=2.5, nominal_duration_sec=2.5,
                is_dead_cycle=True, pressure_bar=180.0, valve_spool_pct=90.0, ram_position_mm=0.0,
                has_anomaly=False, anomaly_type="NONE", injected_delay_sec=0.0
            )
            
            await server.publish_event(ev1)
            await asyncio.sleep(0.1)
            await server.publish_event(ev2)
            await asyncio.sleep(0.1)
            
            assert "container_shift_open" in received_updates
            assert "shear_stroke" in received_updates
            
            await sub.delete()
