import copy
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import swarm
import verify_telemetry


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.config = swarm.load_config(swarm.ROOT / "config.json")

    def test_all_patterns_takeoff_land_and_obey_limits(self):
        for pattern in swarm.PATTERNS:
            config = {**self.config, "pattern": pattern}
            trajectories = swarm.generate_trajectories(config)
            report = swarm.validate_trajectories(config, trajectories)
            self.assertGreaterEqual(report["minimum_separation_m"], 2)
            self.assertLessEqual(report["maximum_speed_m_s"], 5)
            for trajectory in trajectories:
                self.assertAlmostEqual(trajectory[0][3], swarm.GROUND_Z)
                self.assertAlmostEqual(trajectory[-1][3], swarm.GROUND_Z)
                self.assertAlmostEqual(trajectory[0][1], trajectory[-1][1])
                self.assertAlmostEqual(trajectory[0][2], trajectory[-1][2])
                self.assertTrue(all(math.isfinite(value) for row in trajectory for value in row))

    def test_crossing_between_samples_is_rejected(self):
        paths = [[(0, -2, 0, 2, 0), (10, 2, 0, 2, 0)],
                 [(0, 2, 0, 2, 0), (10, -2, 0, 2, 0)]]
        with self.assertRaisesRegex(ValueError, "séparation"):
            swarm.validate_trajectories(self.config, paths)

    def test_overcrowding_and_excessive_speed_are_rejected(self):
        config = {**self.config, "radius": 1}
        with self.assertRaisesRegex(ValueError, "séparation"):
            swarm.validate_trajectories(config, swarm.generate_trajectories(config))
        config = {**self.config, "maximum_speed": 0.01}
        with self.assertRaisesRegex(ValueError, "vitesse"):
            swarm.validate_trajectories(config, swarm.generate_trajectories(config))

    def test_exports_have_ten_local_models_and_matching_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            output = swarm.generate(self.config, folder)
            world = ET.parse(output / "swarm.sdf").getroot().find("world")
            drones = [m for m in world.findall("model") if m.attrib["name"].startswith("drone_")]
            self.assertEqual(len(drones), 10)
            self.assertFalse(world.findall(".//uri"))  # No Fuel/network assets.
            plugin = world.find("plugin[@name='swarm::SwarmPlayback']")
            self.assertEqual(len(plugin.findall("drone")), 10)
            mission = json.loads((output / "mission.json").read_text())
            for index, entry in enumerate(plugin.findall("drone")):
                path = Path(entry.findtext("trajectory"))
                self.assertTrue(path.is_absolute())
                with path.open() as file:
                    samples = [[float(v) for v in row] for row in list(csv.reader(file))[1:]]
                self.assertEqual(samples, mission["trajectories"][index])
            self.assertNotIn("__MISSION_DATA__", (output / "preview.html").read_text())

    def test_fractional_duration_has_exact_endpoint(self):
        config = {**self.config, "duration": 100.023}
        trajectories = swarm.generate_trajectories(config)
        self.assertEqual(trajectories[0][-1][0], config["duration"])
        self.assertAlmostEqual(trajectories[0][-1][3], swarm.GROUND_Z)

    def test_telemetry_verifier_accepts_wrapped_yaw_and_rejects_wrong_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            output = swarm.generate(self.config, folder)
            mission = json.loads((output / "mission.json").read_text())
            rows = []
            for index, track in enumerate(mission["trajectories"]):
                for t, x, y, z, yaw in track[::2]:
                    if t > 14:
                        break
                    rows.append([t, "drone_%02d" % (index + 1), x, y, z,
                                 math.atan2(math.sin(yaw), math.cos(yaw))])
            def write():
                with (output / "telemetry.csv").open("w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(("time", "drone", "x", "y", "z", "yaw"))
                    writer.writerows(rows)
            write()
            verify_telemetry.verify(output)
            rows[10][2] += 1
            write()
            with self.assertRaisesRegex(ValueError, "erreur position"):
                verify_telemetry.verify(output)

    def test_invalid_config_is_rejected(self):
        for key, value in (("drone_count", True), ("sample_hz", 0),
                           ("duration", 10), ("radius", float("nan")),
                           ("minimum_separation", 0.1), ("show_paths", "yes")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                config = copy.deepcopy(self.config)
                config[key] = value
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    swarm.load_config(path)


if __name__ == "__main__":
    unittest.main()
