#!/usr/bin/env python3
"""Check Gazebo's recorded poses against the mission that was actually launched."""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import sys


def verify(output, minimum_duration=14.0, tolerance=0.05):
    mission = json.loads((output / "mission.json").read_text())
    tracks = {"drone_%02d" % (i + 1): track
              for i, track in enumerate(mission["trajectories"])}
    times = [row[0] for row in mission["trajectories"][0]]
    observed = {name: [] for name in tracks}
    maximum_error = 0.0
    with (output / "telemetry.csv").open() as file:
        for row in csv.DictReader(file):
            name = row["drone"]
            if name not in tracks:
                raise ValueError("Drone inattendu : " + name)
            time = float(row["time"])
            pose = [float(row[key]) for key in ("x", "y", "z", "yaw")]
            if not all(math.isfinite(v) for v in [time] + pose):
                raise ValueError("Télémétrie non finie")
            track = tracks[name]
            index = max(0, min(len(times) - 2, bisect.bisect_right(times, time) - 1))
            a, b = track[index], track[index + 1]
            alpha = max(0.0, min(1.0, (time - a[0]) / (b[0] - a[0])))
            expected = [a[k] + alpha * (b[k] - a[k]) for k in (1, 2, 3, 4)]
            error = math.sqrt(sum((pose[k] - expected[k]) ** 2 for k in range(3)))
            yaw_error = abs(math.atan2(math.sin(pose[3] - expected[3]), math.cos(pose[3] - expected[3])))
            maximum_error = max(maximum_error, error)
            if error > tolerance or yaw_error > 0.02:
                raise ValueError("%s à %.2f s : erreur position %.3f m, lacet %.3f rad"
                                 % (name, time, error, yaw_error))
            observed[name].append(time)
    for name, samples in observed.items():
        if not samples or samples[-1] < minimum_duration:
            raise ValueError("Enregistrement absent ou trop court pour " + name)
        if samples[0] > 0.2:
            raise ValueError("Début d'enregistrement manquant pour " + name)
        if any(b <= a or b - a > 0.11 for a, b in zip(samples, samples[1:])):
            raise ValueError("Pas d'enregistrement irrégulier pour " + name)
    print("Gazebo : %s drones vérifiés, %.2f s simulées, erreur maximale %.6f m"
          % (len(observed), min(samples[-1] for samples in observed.values()), maximum_error))


def verify_fire(output, minimum_duration=14.0, tolerance=0.05):
    mission = json.loads((output / "mission.json").read_text())
    if "fire" not in mission:
        return
    tracks = {"fire_%03d" % i: track for i, track in enumerate(mission["fire"]["obstacle_trajectories"])}
    verify_discrete(output, tracks, "fire-telemetry.csv", minimum_duration, tolerance, "Feu")


def verify_water(output, minimum_duration=14.0, tolerance=0.05):
    mission = json.loads((output / "mission.json").read_text())
    if "suppression" not in mission:
        return
    tracks = {"water_%02d" % (i + 1): track
              for i, track in enumerate(mission["suppression"]["stream_trajectories"])}
    verify_discrete(output, tracks, "water-telemetry.csv", minimum_duration, tolerance, "Eau")


def verify_discrete(output, tracks, filename, minimum_duration, tolerance, label):
    times = {name: [sample[0] for sample in track] for name, track in tracks.items()}
    observed = {name: [] for name in tracks}
    max_error = 0.0
    with (output / filename).open() as file:
        for row in csv.DictReader(file):
            name, time = row["model"], float(row["time"])
            if name not in tracks or not math.isfinite(time):
                raise ValueError(label + " : modèle ou temps invalide")
            actor_times = times[name]
            index = max(0, min(len(actor_times) - 1, bisect.bisect_right(actor_times, time) - 1))
            expected = tracks[name][index][1:4]
            actual = [float(row[k]) for k in ("x", "y", "z")]
            if not all(math.isfinite(v) for v in actual):
                raise ValueError(label + " : position non finie")
            error = math.dist(expected, actual)
            max_error = max(max_error, error)
            if error > tolerance:
                raise ValueError("%s %s désynchronisé à %.2f s : %.3f m" % (label, name, time, error))
            observed[name].append(time)
    for name, values in observed.items():
        if not values or values[-1] < minimum_duration or values[0] > 0.2:
            raise ValueError(label + " : enregistrement absent ou trop court pour " + name)
        if any(b <= a or b - a > 0.11 for a, b in zip(values, values[1:])):
            raise ValueError(label + " : pas d'enregistrement irrégulier pour " + name)
    print("Gazebo : %s modèles %s synchronisés, erreur maximale %.6f m" % (len(tracks), label, max_error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--minimum-duration", type=float, default=14.0)
    args = parser.parse_args()
    try:
        verify(args.output, args.minimum_duration)
        verify_fire(args.output, args.minimum_duration)
        verify_water(args.output, args.minimum_duration)
    except (ValueError, OSError, KeyError, csv.Error) as error:
        print("Échec de validation Gazebo : " + str(error), file=sys.stderr)
        sys.exit(1)
