#!/usr/bin/env python3
"""Generate a self-contained Gazebo swarm mission from the current thermal/drone engine."""
import argparse
import colorsys
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from configuration import load_config
from drones.geometry import segment_separation
from drones.spatial import SegmentIndex

ROOT = Path(__file__).resolve().parent
GROUND_Z = 0.3










def validate_trajectories(config, trajectories):
    """Validate the *continuous piecewise linear paths* used by the plugin."""
    if len(trajectories) > 200:
        return validate_large_trajectories(config, trajectories)
    minimum_squared = math.inf
    maximum_speed = 0.0
    for trajectory in trajectories:
        for a, b in zip(trajectory, trajectory[1:]):
            speed = math.sqrt(sum((b[k] - a[k]) ** 2 for k in (1, 2, 3))) / (b[0] - a[0])
            maximum_speed = max(maximum_speed, speed)
    for i, first in enumerate(trajectories):
        for second in trajectories[i + 1:]:
            for a, b, c, d in zip(first, first[1:], second, second[1:]):
                # Relative coordinate extrema bound distance throughout the interval.
                if any((a[k]-c[k]) * (b[k]-d[k]) > 0 and
                       min(abs(a[k]-c[k]), abs(b[k]-d[k])) ** 2 >= minimum_squared
                       for k in (3, 1, 2)):
                    continue
                distance = segment_separation(a[1:4], b[1:4], c[1:4], d[1:4])
                minimum_squared = min(minimum_squared, distance * distance)
    minimum = math.sqrt(minimum_squared)
    if minimum < config["minimum_separation"] - 1e-6:
        raise ValueError("Parcours refusé : séparation %.2f m < %.2f m. Augmenter spacing."
                         % (minimum, config["minimum_separation"]))
    if maximum_speed > config["maximum_speed"] + 1e-6:
        raise ValueError("Parcours refusé : vitesse %.2f m/s > %.2f m/s. Allonger duration."
                         % (maximum_speed, config["maximum_speed"]))
    return {"minimum_separation_m": round(minimum, 6),
            "maximum_speed_m_s": round(maximum_speed, 6),
            "validation": "continuous_linear_interpolation"}


def validate_large_trajectories(config, trajectories):
    """Independently rebuild swept boxes; check every potentially unsafe pair.

    Unchecked pairs have a proven separation >= the configured threshold. The
    reported minimum is therefore a certified lower bound, not an exact minimum.
    """
    limit = config["minimum_separation"]
    closest, maximum_speed, checks = math.inf, 0.0, 0
    length = len(trajectories[0])
    if any(len(track) != length for track in trajectories):
        raise ValueError("Mismatched trajectory lengths")
    for tick in range(length-1):
        index = SegmentIndex(limit)
        segments = []
        for number, track in enumerate(trajectories):
            a, b = track[tick], track[tick+1]
            dt = b[0]-a[0]
            if dt <= 0 or a[0]!=trajectories[0][tick][0] or b[0]!=trajectories[0][tick+1][0]:
                raise ValueError("Invalid or unsynchronized trajectory times")
            start, end = a[1:4], b[1:4]
            if not all(math.isfinite(v) for v in (*start,*end)):
                raise ValueError("Nonfinite trajectory")
            maximum_speed = max(maximum_speed,math.dist(start,end)/dt)
            for other in index.query(start,end):
                distance = segment_separation(start,end,*segments[other])
                closest = min(closest,distance)
                checks += 1
                if distance < limit-1e-6:
                    raise ValueError("Parcours refusé : séparation %.4f m < %.4f m" % (distance,limit))
            index.update(number,start,end)
            segments.append((start,end))
    if maximum_speed > config["maximum_speed"]+1e-6:
        raise ValueError("Parcours refusé : vitesse %.4f m/s" % maximum_speed)
    return {"minimum_separation_m": limit,"minimum_separation_is_lower_bound":True,
            "closest_checked_pair_m":closest if math.isfinite(closest) else None,
            "maximum_speed_m_s":round(maximum_speed,6),"pair_checks":checks,
            "validation":"continuous_linear_interpolation_spatial_broad_phase"}


def element(parent, tag, text=None, **attributes):
    child = ET.SubElement(parent, tag, attributes)
    if text is not None:
        child.text = str(text)
    return child


def color(index, count):
    return (*colorsys.hsv_to_rgb(index / count, 0.72, 0.95), 1.0)


def visual(link, name, pose, shape, dimensions, rgba):
    item = element(link, "visual", name=name)
    element(item, "pose", " ".join(str(v) for v in pose))
    geometry = element(element(item, "geometry"), shape)
    for key, value in dimensions.items():
        element(geometry, key, value)
    material = element(item, "material")
    value = " ".join(str(v) for v in rgba)
    element(material, "ambient", value)
    element(material, "diffuse", value)
    return item


def world_document(config, trajectories, output, scenario=None):
    sdf = ET.Element("sdf", version="1.9")
    bound = max(50.0, config.get("navigation", {}).get("world_bound", 30.0))
    size = 2 * bound
    grid_bound = int(config.get("navigation", {}).get("world_bound", 30))
    scale = config.get("suppression", {}).get("drone_radius", 0.5) / 0.5
    world = element(sdf, "world", name="swarm_beta")
    physics = element(world, "physics", name="default", type="ignored")
    element(physics, "max_step_size", "0.01")
    element(physics, "real_time_factor", "1.0")
    gui = element(world, "gui", fullscreen="false")
    view = element(gui, "plugin", filename="MinimalScene", name="3D View")
    settings = element(view, "gz-gui")
    element(settings, "title", "Essaim de drones")
    element(settings, "property", "false", key="showTitleBar", type="bool")
    element(settings, "property", "docked", key="state", type="string")
    element(view, "engine", "ogre2")
    if sys.platform == "darwin":
        element(view, "render_system", "metal")
    element(view, "scene", "scene")
    element(view, "ambient_light", "0.5 0.5 0.5")
    element(view, "background_color", "0.06 0.08 0.12")
    camera_scale = 3 if bound > 50 else 1
    element(view, "camera_pose", "%s %s %s 0 %s 2.28" % (
        bound*.6*camera_scale, -bound*.7*camera_scale, bound*.56*camera_scale,
        0.5 if camera_scale > 1 else 0.45))
    for filename in ("GzSceneManager", "InteractiveViewControl", "CameraTracking"):
        tool = element(gui, "plugin", filename=filename, name=filename)
        settings = element(tool, "gz-gui")
        for key, value, kind in (("state", "floating", "string"),
                                 ("showTitleBar", "false", "bool"),
                                 ("width", "5", "double"), ("height", "5", "double")):
            element(settings, "property", value, key=key, type=kind)
    for filename, side in (("WorldControl", "left"), ("WorldStats", "right")):
        tool = element(gui, "plugin", filename=filename, name=filename)
        settings = element(tool, "gz-gui")
        for key, value, kind in (("state", "floating", "string"),
                                 ("showTitleBar", "false", "bool"),
                                 ("width", "280", "double"), ("height", "90", "double")):
            element(settings, "property", value, key=key, type=kind)
        anchors = element(settings, "anchors", target="Essaim de drones")
        element(anchors, "line", own=side, target=side)
        element(anchors, "line", own="bottom", target="bottom")
        keys = ("play_pause", "step", "use_event") if filename == "WorldControl" else (
            "sim_time", "real_time_factor", "iterations")
        for key in keys:
            element(tool, key, "true")
        if filename == "WorldControl":
            element(tool, "start_paused", "false")
    systems = ("UserCommands", "SceneBroadcaster") if config["drone_count"]>200 else (
        "Physics", "UserCommands", "SceneBroadcaster")
    for system in systems:
        filename = {"Physics": "physics", "UserCommands": "user-commands",
                    "SceneBroadcaster": "scene-broadcaster"}[system]
        element(world, "plugin", filename="gz-sim-" + filename + "-system",
                name="gz::sim::systems::" + system)
    scene = element(world, "scene")
    element(scene, "ambient", "0.5 0.5 0.5 1")
    element(scene, "background", "0.06 0.08 0.12 1")
    light = element(world, "light", name="sun", type="directional")
    element(light, "pose", "0 0 30 0 0 0")
    element(light, "direction", "-0.5 0.2 -1")
    element(light, "diffuse", "0.85 0.85 0.85 1")
    element(light, "cast_shadows", "true")
    ground = element(world, "model", name="ground")
    element(ground, "static", "true")
    link = element(ground, "link", name="ground_link")
    visual(link, "ground", (0, 0, -0.05, 0, 0, 0), "box",
           {"size": "%s %s 0.1" % (size, size)}, (0.14, 0.19, 0.21, 1))
    collision = element(link, "collision", name="ground_collision")
    element(element(element(collision, "geometry"), "box"), "size", "%s %s 0.1" % (size, size))
    for axis in range(2):
        for grid in range(-grid_bound, grid_bound + 1, max(5, grid_bound // 15)):
            pose = (grid, 0, 0.003, 0, 0, 0) if axis == 0 else (0, grid, 0.003, 0, 0, 0)
            visual(link, "grid_%s_%s" % (axis, grid), pose, "box",
                   {"size": "0.025 %s 0.005" % (2*grid_bound) if axis == 0 else "%s 0.025 0.005" % (2*grid_bound)},
                   (0.3, 0.36, 0.38, 1))
    plugin = element(world, "plugin", filename="SwarmPlayback", name="swarm::SwarmPlayback")
    element(plugin, "direct_pose", "true" if config["drone_count"]>200 else "false")
    element(plugin, "telemetry", output / "telemetry.csv")
    if scenario is not None:
        element(plugin, "fire_telemetry", output / "fire-telemetry.csv")
        target = element(world, "model", name="target")
        element(target, "static", "true")
        target_link = element(target, "link", name="target_link")
        for index, (gx, gy) in enumerate(scenario["navigation"]["goal_positions"]):
            visual(target_link, "landing_%s" % index, (gx, gy, 0.02, 0, 0, 0), "cylinder",
                   {"radius": "0.55", "length": "0.03"}, color(index, config["drone_count"]))
        for index, track in enumerate(scenario["fire"]["obstacle_trajectories"]):
            name = "fire_%03d" % index
            entry = element(plugin, "obstacle")
            element(entry, "name", name)
            element(entry, "trajectory", output / "obstacles" / (name + ".csv"))
            model = element(world, "model", name=name)
            element(model, "static", "true")
            x, y, z, _ = track[0][1:]
            element(model, "pose", "%s %s %s 0 0 0" % (x, y, z))
            body = element(model, "link", name="fire_link")
            height, size = config["fire"]["height"], scenario["fire"].get("render_cell_size",config["fire"]["cell_size"])
            flame = visual(body, "danger_volume", (0, 0, 0, 0, 0, 0), "box",
                           {"size": "%s %s %s" % (size, size, height)}, (1, 0.2, 0.03, 0.22))
            element(flame.find("material"), "emissive", "0.8 0.1 0 1")
            visual(body, "flame_base", (0, 0, -height / 2 + 0.5, 0, 0, 0), "cylinder",
                   {"radius": str(size * 0.42), "length": "1.0"}, (1, 0.4, 0.02, 1))
        if "suppression" in scenario:
            element(plugin, "water_telemetry", output / "water-telemetry.csv")
            refill_points = scenario["suppression"]["refill_points"]
            for index, (sx, sy, sz) in enumerate(refill_points):
                name = "refill_station" if len(refill_points) == 1 else "refill_station_%02d" % (index + 1)
                station = element(world, "model", name=name)
                element(station, "static", "true")
                station_link = element(station, "link", name="station_link")
                visual(station_link, "reservoir", (sx, sy, 0.25, 0, 0, 0), "cylinder",
                       {"radius": "1.4", "length": "0.5"}, (0.1, 0.5, 1, 1))
                visual(station_link, "refill_pad", (sx, sy, sz - 0.3, 0, 0, 0), "cylinder",
                       {"radius": str(.6*scale), "length": "0.05"}, (0.5, 0.85, 1, 1))
            for i, stream in enumerate(scenario["suppression"]["stream_trajectories"]):
                name = "water_%02d" % (i + 1)
                entry = element(plugin, "water")
                element(entry, "name", name)
                element(entry, "trajectory", output / "water" / (name + ".csv"))
                model = element(world, "model", name=name)
                element(model, "static", "true")
                element(model, "pose", "%s %s -50 0 0 0" % (stream[0][1], stream[0][2]))
                body = element(model, "link", name="stream_link")
                lane = config["suppression"]["altitude_lanes"][i % len(config["suppression"]["altitude_lanes"])]
                visual(body, "water_stream", (0, 0, 0, 0, 0, 0), "cylinder",
                       {"radius": "0.1", "length": str(lane)}, (0.15, 0.6, 1, 0.7))
    for index, trajectory in enumerate(trajectories):
        name = "drone_%02d" % (index + 1)
        entry = element(plugin, "drone")
        element(entry, "name", name)
        element(entry, "trajectory", output / "trajectories" / (name + ".csv"))
        model = element(world, "model", name=name)
        # Static models follow explicit poses. Gravity/collisions do not drive them.
        element(model, "static", "true")
        x, y, z, yaw = trajectory[0][1:]
        element(model, "pose", "%s %s %s 0 0 %s" % (x, y, z, yaw))
        body = element(model, "link", name="base_link")
        rgba = color(index, config["drone_count"])
        visual(body, "body", (0, 0, 0, 0, 0, 0), "box", {"size": "%s %s %s" % (.32*scale, .22*scale, .12*scale)}, rgba)
        visual(body, "nose", (0.19*scale, 0, 0, 0, 0, 0), "box", {"size": "%s %s %s" % (.08*scale, .1*scale, .07*scale)}, (1, 1, 1, 1))
        for arm in (-1, 1):
            visual(body, "arm_%s" % arm, (0, 0, 0, 0, 0, arm * math.pi / 4),
                   "box", {"size": "%s %s %s" % (.75*scale, .035*scale, .035*scale)}, (0.2, 0.22, 0.25, 1))
        for rotor, (dx, dy) in enumerate(((-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25))):
            visual(body, "rotor_%s" % rotor, (dx*scale, dy*scale, .07*scale, 0, 0, 0), "cylinder",
                   {"radius": str(.12*scale), "length": str(.012*scale)}, rgba)
        visual(link, "pad_%s" % index, (x, y, 0.02, 0, 0, 0), "cylinder",
               {"radius": "0.55", "length": "0.02"}, rgba)
        if config["show_paths"]:
            paths = element(world, "model", name="path_%02d" % (index + 1))
            element(paths, "static", "true")
            path_link = element(paths, "link", name="path_link")
            stride = max(1, (len(trajectory) - 1) // 48)
            points = trajectory[::stride]
            if points[-1] != trajectory[-1]:
                points.append(trajectory[-1])
            for segment, (a, b) in enumerate(zip(points, points[1:])):
                dx, dy, dz = (b[k] - a[k] for k in (1, 2, 3))
                length = math.sqrt(dx * dx + dy * dy + dz * dz)
                if length < 1e-6:
                    continue
                pose = tuple((a[k] + b[k]) / 2 for k in (1, 2, 3)) + (
                    0, math.acos(max(-1, min(1, dz / length))), math.atan2(dy, dx))
                visual(path_link, "segment_%s" % segment, pose, "cylinder",
                       {"radius": "0.015", "length": str(length)}, rgba)
    return sdf


def generate(config, output):
    from simulation import mission
    output = Path(output).resolve()
    scenario = mission.generate_mission(config)
    trajectories = scenario['trajectories']
    metrics = validate_trajectories(config, trajectories)
    mission.validate_paths(config, scenario)
    metrics.update(scenario['suppression']['metrics'])
    metrics['fire_avoidance_validated'] = True
    (output / "trajectories").mkdir(parents=True, exist_ok=True)
    for index, samples in enumerate(trajectories):
        with (output / "trajectories" / ("drone_%02d.csv" % (index + 1))).open("w", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(("time", "x", "y", "z", "yaw"))
            writer.writerows(samples)
    if scenario:
        (output / "obstacles").mkdir(exist_ok=True)
        for index, samples in enumerate(scenario["fire"]["obstacle_trajectories"]):
            with (output / "obstacles" / ("fire_%03d.csv" % index)).open("w", newline="") as file:
                writer = csv.writer(file, lineterminator="\n")
                writer.writerow(("time", "x", "y", "z", "yaw"))
                writer.writerows(samples)
        if "suppression" in scenario:
            (output / "water").mkdir(exist_ok=True)
            for i, samples in enumerate(scenario["suppression"]["stream_trajectories"]):
                with (output / "water" / ("water_%02d.csv" % (i + 1))).open("w", newline="") as file:
                    writer = csv.writer(file, lineterminator="\n")
                    writer.writerow(("time", "x", "y", "z", "yaw"))
                    writer.writerows(samples)
    ET.ElementTree(world_document(config, trajectories, output, scenario)).write(
        output / "swarm.sdf", encoding="utf-8", xml_declaration=True)
    data = {"config": config, "metrics": metrics, "trajectories": trajectories}
    if scenario:
        data.update({"fire": scenario["fire"], "navigation": scenario["navigation"]})
        if "suppression" in scenario:
            data["suppression"] = scenario["suppression"]
            (output / "water-drops.csv").write_text("time,drone,x,y,z,litres\n" + "".join(
                "%s,%s,%s,%s,%s,%s\n" % (drop["time"], drop["drone"], *drop["position"], drop["litres"])
                for drop in scenario["suppression"]["drops"]))
            (output / "controller-decisions.json").write_text(json.dumps(scenario["suppression"]["decisions"], separators=(",", ":")))
    (output / "mission.json").write_text(json.dumps(data, separators=(",", ":")))
    (output / "report.json").write_text(json.dumps({"config": config, "metrics": metrics}, indent=2) + "\n")
    template = (ROOT / "preview.html").read_text()
    preview_data = dict(data)
    if "suppression" in data:
        # The renderer uses observations and states; native tracks and policy
        # audit logs remain in mission.json without making the browser parse them.
        preview_data["fire"] = {key: value for key, value in data["fire"].items() if key != "obstacle_trajectories"}
        preview_data["suppression"] = {key: value for key, value in data["suppression"].items()
                                       if key in ("states", "metrics", "refill_points", "stream_events")}
        if config["drone_count"]>200:
            preview_data["trajectories"] = [[[round(v,4) for v in row] for row in track] for track in trajectories]
    (output / "preview.html").write_text(template.replace("__MISSION_DATA__", json.dumps(preview_data, separators=(",", ":"))))
    print("%s : %s drones, %s, %.1f s" % (output, config["drone_count"], config["pattern"], config["duration"]))
    print("Séparation minimale : %.2f m ; vitesse maximale : %.2f m/s" %
          (metrics["minimum_separation_m"], metrics["maximum_speed_m_s"]))
    if scenario and "suppression" in scenario:
        print("Water: %.1f L dropped; %d refills; %d fire cells extinguished by water; %.1f%% observed edges wetted"
              % (metrics["water_dropped_l"], metrics["refills_completed"], metrics["extinguished_by_water"],
                 metrics["perimeter_coverage_fraction"] * 100))
    elif scenario:
        print("Feu seed=%s ; %s recalculs ; cible atteinte à %.2f s" %
              (config["fire"]["seed"], metrics["replans"], metrics["arrival_time"]))
    return output


def doctor():
    missing = []
    for command in ("python3", "cmake", "gz"):
        found = shutil.which(command)
        print(command + ": " + (found or "absent"))
        if not found:
            missing.append(command)
    if shutil.which("gz"):
        result = subprocess.run(["gz", "sim", "--versions"], text=True, capture_output=True)
        version = result.stdout.strip()
        print("Gazebo Sim : " + (version or result.stderr.strip()))
        if result.returncode or not any(part.startswith("8.") for part in version.split()):
            missing.append("gz-sim8 (Harmonic)")
    print("Plugin : " + ("compilé" if any((ROOT / "build").glob("*SwarmPlayback.*")) else "à compiler"))
    if missing:
        print("Dépendances manquantes : " + ", ".join(missing))
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "doctor"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ppo-1000-fast.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            return doctor()
        config = load_config(args.config)
        generate(config, args.output)
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print("Erreur : " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
