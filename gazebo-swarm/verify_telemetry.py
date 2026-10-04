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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--minimum-duration", type=float, default=14.0)
    args = parser.parse_args()
    try:
        verify(args.output, args.minimum_duration)
    except (ValueError, OSError, KeyError) as error:
        print("Échec de validation Gazebo : " + str(error), file=sys.stderr)
        sys.exit(1)
