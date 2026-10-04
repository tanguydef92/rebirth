#!/usr/bin/env python3
"""Generate a self-contained Gazebo swarm mission using Python's stdlib only."""
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

ROOT = Path(__file__).resolve().parent
PATTERNS = ("circle", "helix", "sweep")
GROUND_Z = 0.3


def smooth(u):
    u = max(0.0, min(1.0, u))
    return u * u * u * (10 + u * (-15 + 6 * u))


def load_config(path):
    with Path(path).open() as file:
        config = json.load(file)
    required = json.loads((ROOT / "config.json").read_text())
    if set(config) != set(required):
        raise ValueError("Les clés du fichier doivent correspondre à config.json")
    if type(config["drone_count"]) is not int or not 2 <= config["drone_count"] <= 50:
        raise ValueError("drone_count doit être un entier entre 2 et 50")
    if config["pattern"] not in PATTERNS:
        raise ValueError("pattern doit être circle, helix ou sweep")
    if type(config["show_paths"]) is not bool:
        raise ValueError("show_paths doit être un booléen")
    for key in required.keys() - {"drone_count", "pattern", "show_paths"}:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(key + " doit être numérique")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(key + " doit être positif et fini")
    if not 5 <= config["sample_hz"] <= 100:
        raise ValueError("sample_hz doit être entre 5 et 100")
    if config["duration"] > 600:
        raise ValueError("La durée maximale de la bêta est de 600 secondes")
    if config["duration"] <= config["takeoff_duration"] + config["landing_duration"]:
        raise ValueError("La durée doit laisser du temps pour la mission")
    if config["altitude"] <= GROUND_Z:
        raise ValueError("L'altitude doit être supérieure à 0.3 m")
    if config["minimum_separation"] < 1.0:
        raise ValueError("minimum_separation doit être au moins 1 m (taille des drones)")
    return config


def position(config, index, time):
    duration = config["duration"]
    takeoff = config["takeoff_duration"]
    landing = config["landing_duration"]
    mission = duration - takeoff - landing
    phase = smooth((time - takeoff) / mission)
    altitude = config["altitude"]
    if time < takeoff:
        altitude = GROUND_Z + (altitude - GROUND_Z) * smooth(time / takeoff)
    elif time > duration - landing:
        altitude = GROUND_Z + (altitude - GROUND_Z) * smooth((duration - time) / landing)
    angle = 2 * math.pi * index / config["drone_count"]
    if config["pattern"] in ("circle", "helix"):
        angle += 2 * math.pi * phase
        x = config["radius"] * math.cos(angle)
        y = config["radius"] * math.sin(angle)
        if config["pattern"] == "helix":
            altitude += config["helix_height"] * math.sin(math.pi * phase) ** 2
        yaw = angle + math.pi / 2
    else:
        columns = math.ceil(math.sqrt(config["drone_count"]))
        rows = math.ceil(config["drone_count"] / columns)
        x = (index % columns - (columns - 1) / 2) * config["spacing"]
        y = (index // columns - (rows - 1) / 2) * config["spacing"]
        x += config["radius"] * math.sin(2 * math.pi * phase)
        y += 0.5 * config["radius"] * math.sin(4 * math.pi * phase)
        yaw = 0.0
    return (x, y, altitude, yaw)


def generate_trajectories(config):
    # The last sample always lands exactly at duration, even for fractional times.
    intervals = math.ceil(config["duration"] * config["sample_hz"])
    times = [i * config["duration"] / intervals for i in range(intervals + 1)]
    return [[(t,) + position(config, i, t) for t in times]
            for i in range(config["drone_count"])]


def validate_trajectories(config, trajectories):
    """Validate the *continuous piecewise linear paths* used by the plugin."""
    minimum_squared = math.inf
    maximum_speed = 0.0
    for trajectory in trajectories:
        for a, b in zip(trajectory, trajectory[1:]):
            speed = math.sqrt(sum((b[k] - a[k]) ** 2 for k in (1, 2, 3))) / (b[0] - a[0])
            maximum_speed = max(maximum_speed, speed)
    for i, first in enumerate(trajectories):
        for second in trajectories[i + 1:]:
            for a, b, c, d in zip(first, first[1:], second, second[1:]):
                offset = [a[k] - c[k] for k in (1, 2, 3)]
                delta = [(b[k] - d[k]) - offset[k - 1] for k in (1, 2, 3)]
                norm = sum(v * v for v in delta)
                alpha = max(0.0, min(1.0, -sum(u * v for u, v in zip(offset, delta)) / norm)) if norm else 0.0
                distance_squared = sum((u + alpha * v) ** 2 for u, v in zip(offset, delta))
                minimum_squared = min(minimum_squared, distance_squared)
    minimum = math.sqrt(minimum_squared)
    if minimum < config["minimum_separation"] - 1e-6:
        raise ValueError("Parcours refusé : séparation %.2f m < %.2f m. Augmenter radius ou spacing."
                         % (minimum, config["minimum_separation"]))
    if maximum_speed > config["maximum_speed"] + 1e-6:
        raise ValueError("Parcours refusé : vitesse %.2f m/s > %.2f m/s. Allonger duration."
                         % (maximum_speed, config["maximum_speed"]))
    return {"minimum_separation_m": round(minimum, 6),
            "maximum_speed_m_s": round(maximum_speed, 6),
            "validation": "continuous_linear_interpolation"}


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


def world_document(config, trajectories, output):
    sdf = ET.Element("sdf", version="1.9")
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
    element(view, "camera_pose", "30 -35 28 0 0.45 2.28")
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
    for system in ("Physics", "UserCommands", "SceneBroadcaster"):
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
           {"size": "100 100 0.1"}, (0.14, 0.19, 0.21, 1))
    collision = element(link, "collision", name="ground_collision")
    element(element(element(collision, "geometry"), "box"), "size", "100 100 0.1")
    for axis in range(2):
        for grid in range(-30, 31, 5):
            pose = (grid, 0, 0.003, 0, 0, 0) if axis == 0 else (0, grid, 0.003, 0, 0, 0)
            visual(link, "grid_%s_%s" % (axis, grid), pose, "box",
                   {"size": "0.025 60 0.005" if axis == 0 else "60 0.025 0.005"},
                   (0.3, 0.36, 0.38, 1))
    plugin = element(world, "plugin", filename="SwarmPlayback", name="swarm::SwarmPlayback")
    element(plugin, "telemetry", output / "telemetry.csv")
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
        visual(body, "body", (0, 0, 0, 0, 0, 0), "box", {"size": "0.32 0.22 0.12"}, rgba)
        visual(body, "nose", (0.19, 0, 0, 0, 0, 0), "box", {"size": "0.08 0.1 0.07"}, (1, 1, 1, 1))
        for arm in (-1, 1):
            visual(body, "arm_%s" % arm, (0, 0, 0, 0, 0, arm * math.pi / 4),
                   "box", {"size": "0.75 0.035 0.035"}, (0.2, 0.22, 0.25, 1))
        for rotor, (dx, dy) in enumerate(((-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25))):
            visual(body, "rotor_%s" % rotor, (dx, dy, 0.07, 0, 0, 0), "cylinder",
                   {"radius": "0.12", "length": "0.012"}, rgba)
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
    output = Path(output).resolve()
    trajectories = generate_trajectories(config)
    metrics = validate_trajectories(config, trajectories)
    (output / "trajectories").mkdir(parents=True, exist_ok=True)
    for index, samples in enumerate(trajectories):
        with (output / "trajectories" / ("drone_%02d.csv" % (index + 1))).open("w", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(("time", "x", "y", "z", "yaw"))
            writer.writerows(samples)
    ET.ElementTree(world_document(config, trajectories, output)).write(
        output / "swarm.sdf", encoding="utf-8", xml_declaration=True)
    data = {"config": config, "metrics": metrics, "trajectories": trajectories}
    (output / "mission.json").write_text(json.dumps(data, separators=(",", ":")))
    (output / "report.json").write_text(json.dumps({"config": config, "metrics": metrics}, indent=2) + "\n")
    template = (ROOT / "preview.html").read_text()
    (output / "preview.html").write_text(template.replace("__MISSION_DATA__", json.dumps(data, separators=(",", ":"))))
    print("%s : %s drones, %s, %.1f s" % (output, config["drone_count"], config["pattern"], config["duration"]))
    print("Séparation minimale : %.2f m ; vitesse maximale : %.2f m/s" %
          (metrics["minimum_separation_m"], metrics["maximum_speed_m_s"]))
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
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--pattern", choices=PATTERNS)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            return doctor()
        config = load_config(args.config)
        if args.pattern:
            config["pattern"] = args.pattern
        generate(config, args.output)
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print("Erreur : " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
