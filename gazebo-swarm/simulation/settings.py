"""Validate current drone navigation, station and water-service settings."""
import math
from configuration import finite_number,validate_fire_config
from drones.controller import SwarmController

DRONE_RADIUS = .5


def validate_config(config):
    nav, policy = config.get("navigation"), config.get("suppression")
    nav_keys = {"start", "cruise_speed", "vertical_speed", "world_bound", "clearance"}
    nav_optional = {"descent_speed", "start_layout", "corner_offset"}
    if not isinstance(nav, dict) or not nav_keys <= set(nav) or set(nav) - nav_keys - nav_optional:
        raise ValueError("Suppression navigation keys: see configs/ppo-1000-fast.json")
    policy_keys = {
            "tank_capacity_l", "water_to_extinguish_l",
            "drop_radius", "refill_point", "holding_radius", "altitude_lanes", "decision_interval",
            "return_duration", "drop_duration", "refill_duration"}
    optional = {"refill_points", "water_application_l_m2", "water_payload_kg",
                "water_density_kg_l", "perimeter_only", "drone_radius", "refill_layout", "allocation_model",
                "ppo_checkpoint", "state_interval", "compact_log", "cache_policy_actions", "record_decisions"}
    if not isinstance(policy, dict) or not policy_keys <= set(policy) or set(policy) - policy_keys - optional:
        raise ValueError("Suppression policy keys: see configs/ppo-1000-fast.json")
    validate_fire_config(config['fire'],config['duration'],nav['world_bound'])
    for key in ('cruise_speed','world_bound','clearance'):
        finite_number(nav[key],key)
    if nav['world_bound']>500 or nav['cruise_speed']>config['maximum_speed']:
        raise ValueError('World bound or cruise speed exceeds its limit')
    if "refill_layout" in policy and policy["refill_layout"] != "homes":
        raise ValueError("refill_layout must be homes")
    if policy.get("allocation_model") not in ("ppo","uniform"):
        raise ValueError("allocation_model must be ppo or uniform")
    if policy.get("allocation_model") == "ppo" and (not isinstance(policy.get("ppo_checkpoint"),str) or not policy["ppo_checkpoint"]):
        raise ValueError("PPO requires ppo_checkpoint; train with train_ppo.py first")
    if "compact_log" in policy and type(policy["compact_log"]) is not bool:
        raise ValueError("compact_log must be boolean")
    for key in ("cache_policy_actions", "record_decisions"):
        if key in policy and type(policy[key]) is not bool:raise ValueError(key+" must be boolean")
    finite_number(nav["vertical_speed"], "vertical_speed")
    if nav["vertical_speed"] > config["maximum_speed"]:
        raise ValueError("vertical_speed exceeds maximum_speed")
    if "descent_speed" in nav:
        finite_number(nav["descent_speed"], "descent_speed")
        if nav["descent_speed"] > config["maximum_speed"]:
            raise ValueError("descent_speed exceeds maximum_speed")
    if "start_layout" in nav:
        if nav["start_layout"] != "corners":
            raise ValueError("start_layout must be corners")
        finite_number(nav.get("corner_offset"), "corner_offset")
    if "perimeter_only" in policy and type(policy["perimeter_only"]) is not bool:
        raise ValueError("perimeter_only must be a boolean")
    for key in policy.keys() - {"refill_point", "refill_points", "altitude_lanes", "perimeter_only",
                                "refill_layout", "allocation_model", "ppo_checkpoint", "compact_log", "cache_policy_actions", "record_decisions"}:
        finite_number(policy[key], key, positive=True)
        if policy[key] < 0:
            raise ValueError(key + " must be non-negative")
    if "water_application_l_m2" in policy:
        policy["water_to_extinguish_l"] = policy["water_application_l_m2"] * config["fire"]["cell_size"] ** 2
    if "water_payload_kg" in policy:
        density = policy.get("water_density_kg_l", 1.0)
        if abs(policy["tank_capacity_l"] * density - policy["water_payload_kg"]) > 1e-6:
            raise ValueError("tank_capacity_l * water_density_kg_l must equal water_payload_kg")
    if config["fire"].get("model") == "lagrangian":
        from fire.particles import DEFAULT_PHYSICS
        density = config["fire"].get("physics",{}).get("water_density_kg_l",DEFAULT_PHYSICS["water_density_kg_l"])
        if abs(density-policy.get("water_density_kg_l",1.0)) > 1e-9:
            raise ValueError("Thermal fire and drone payload must use the same water density")
    if 2 * policy.get("drone_radius", DRONE_RADIUS) > config["minimum_separation"]:
        raise ValueError("minimum_separation must exceed the drone diameter")
    if not 0.2 <= policy["decision_interval"] <= 10:
        raise ValueError("decision_interval must be between 0.2 and 10 seconds")
    if policy["return_duration"] + config["takeoff_duration"] + config["landing_duration"] >= config["duration"]:
        raise ValueError("No time remains for suppression")
    lanes = policy["altitude_lanes"]
    if not isinstance(lanes, list) or not lanes:
        raise ValueError("altitude_lanes must be a non-empty list")
    for lane in lanes:
        finite_number(lane, "altitude_lanes")
        if not config["fire"]["height"] + nav["clearance"] + policy.get("drone_radius", DRONE_RADIUS) < lane <= 120:
            raise ValueError("Flight lanes must clear fire height + clearance + drone radius, and be <= 120 m")
    for i, lane in enumerate(lanes):
        if any(abs(lane - other) < config["minimum_separation"] for other in lanes[i + 1:]):
            raise ValueError("Altitude lanes must be separated by minimum_separation")
    station = policy["refill_point"]
    if not isinstance(station, list) or len(station) != 3:
        raise ValueError("refill_point must contain [x, y, z]")
    for value in station:
        finite_number(value, "refill_point", positive=False)
    if station[2] < 0.3 or station[2] >= min(lanes):
        raise ValueError("Refill height must be >= 0.3 m and below flight lanes")
    if config["spacing"] < config["minimum_separation"]:
        raise ValueError("Initial spacing must respect minimum_separation")
    points = policy.get("refill_points", [station])
    if "refill_points" in policy and (not isinstance(points, list) or len(points) != config["drone_count"]):
        raise ValueError("refill_points requires one station per drone")
    for point in points:
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError("Each refill point must contain [x, y, z]")
        for value in point:
            finite_number(value, "refill_points", positive=False)
        if not 0.3 <= point[2] < min(lanes):
            raise ValueError("Refill heights must be >= 0.3 m and below flight lanes")
    controller = SwarmController(config)
    for i, point in enumerate(controller.stations):
        if any(math.dist(point, other) < config["minimum_separation"] for other in controller.stations[i + 1:]):
            raise ValueError("Refill stations must respect minimum_separation")
    radius = policy.get("drone_radius", DRONE_RADIUS)
    size = config["fire"]["cell_size"]
    half = size / 2 + nav["clearance"] + radius
    x0,x1,y0,y1 = config["fire"]["domain"]
    for point in controller.stations + [drone.home for drone in controller.drones] + [drone.holding for drone in controller.drones]:
        if any(abs(point[k]) > nav["world_bound"] for k in (0, 1)):
            raise ValueError("Station, homes and holding positions must be within world_bound")
        if point[2] <= config["fire"]["height"] + nav["clearance"] + radius:
            if x0*size-half <= point[0] <= x1*size+half and y0*size-half <= point[1] <= y1*size+half:
                raise ValueError("Low station/home positions must be outside the entire fire domain and clearance")
    for i, drone in enumerate(controller.drones):
        if any(math.dist(drone.home, other.home) < config["minimum_separation"]
               for other in controller.drones[i + 1:]):
            raise ValueError("Corner starts overlap; increase corner_offset or reduce spacing")
    for i, drone in enumerate(controller.drones):
        if any(math.dist(drone.holding, other.holding) < config["minimum_separation"]
               for other in controller.drones[i + 1:]):
            raise ValueError("Holding positions are too close; increase holding_radius")
