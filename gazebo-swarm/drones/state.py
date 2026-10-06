"""Individual drone state, shared by training and simulation."""
from dataclasses import dataclass

@dataclass
class Drone:
    number: int
    position: tuple
    home: tuple
    lane: float
    holding: tuple
    water_l: float
    mode: str = "takeoff"
    target: object = None
    goal: object = None
    blocked: bool = False
    station_index: object = None
    service_started: object = None
    service_water: float = 0.0
    drop_until: object = None


def fleet_summary(drones, capacity):
    operational = [drone for drone in drones if drone.mode in ("attack", "drop", "idle") and drone.water_l > 0]
    return {"water_fraction":sum(d.water_l for d in drones)/(len(drones)*capacity),
            "ready_fraction":len(operational)/len(drones),
            "available_water_l":sum(d.water_l for d in operational)}
