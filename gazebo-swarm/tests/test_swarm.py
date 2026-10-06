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
        self.config = swarm.load_config(swarm.ROOT / "configs/ppo-100-fast.json")
        self.config.update(drone_count=8,duration=110,takeoff_duration=20,landing_duration=20)
        self.config['navigation'].update(start=[-60,-60],corner_offset=60)
        self.config['suppression'].update(return_duration=40,altitude_lanes=[20,26],allocation_model='uniform')
        self.config['fire'].update(domain=[-10,10,-10,10],initial_rectangle=[-3,3,-3,3],terrain_regions=[])


    def test_crossing_between_samples_is_rejected(self):
        paths = [[(0, -2, 0, 2, 0), (10, 2, 0, 2, 0)],
                 [(0, 2, 0, 2, 0), (10, -2, 0, 2, 0)]]
        with self.assertRaisesRegex(ValueError, "séparation"):
            swarm.validate_trajectories(self.config, paths)


    def test_exports_have_current_drone_models_and_matching_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            output = swarm.generate(self.config, folder)
            world = ET.parse(output / "swarm.sdf").getroot().find("world")
            drones = [m for m in world.findall("model") if m.attrib["name"].startswith("drone_")]
            self.assertEqual(len(drones), 8)
            self.assertFalse(world.findall(".//uri"))  # No Fuel/network assets.
            plugin = world.find("plugin[@name='swarm::SwarmPlayback']")
            self.assertEqual(len(plugin.findall("drone")), 8)
            mission = json.loads((output / "mission.json").read_text())
            for index, entry in enumerate(plugin.findall("drone")):
                path = Path(entry.findtext("trajectory"))
                self.assertTrue(path.is_absolute())
                with path.open() as file:
                    samples = [[float(v) for v in row] for row in list(csv.reader(file))[1:]]
                self.assertEqual(samples, mission["trajectories"][index])
            self.assertNotIn("__MISSION_DATA__", (output / "preview.html").read_text())


    def test_telemetry_verifier_accepts_wrapped_yaw_and_rejects_wrong_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            output = swarm.generate(self.config, folder)
            mission = json.loads((output / "mission.json").read_text())
            rows = []
            for index, track in enumerate(mission["trajectories"]):
                for tick in range(141):
                    t=tick/10
                    k=min(len(track)-2,int(t*self.config['sample_hz']))
                    a,b=track[k],track[k+1];u=(t-a[0])/(b[0]-a[0])
                    x,y,z,yaw=[a[j]+u*(b[j]-a[j]) for j in (1,2,3,4)]
                    rows.append([t,"drone_%02d"%(index+1),x,y,z,math.atan2(math.sin(yaw),math.cos(yaw))])
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
                           ("duration", 10), ("maximum_speed", float("nan")),
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
