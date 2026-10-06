"""Swarm policy and movement decisions, independent of Gazebo and the fire RNG.

Modeling extension points: allocate_defense() for target assignments and
next_direction() for movements. The PPO observation contract is in
drones/policy.py; training reward and transitions are in training/individual_env.py.
step() receives a *current observation* only. fire_geometry contains polygons,
exposed edges, estimated spread and cooling demand for that move, never future states.
"""
from drones.state import Drone, fleet_summary
from drones.navigation import DroneNavigator
import math
from pathlib import Path
from drones.spatial import SegmentIndex
from drones.policy import PPOPolicy, observation_vector, uniform_fractions, SECTORS

from drones.geometry import formation_offsets


class SwarmController:
    @staticmethod
    def edge_key(edge):
        # Canonical physical face midpoint: opposite orientations are one edge.
        x, y, dx, dy = edge
        return (2*x + dx, 2*y + dy)

    def __init__(self, config):
        self.config = config
        self.policy = config["suppression"]
        self.nav = config["navigation"]
        self.station = tuple(self.policy["refill_point"])
        self.stations = [tuple(point) for point in self.policy.get("refill_points", [self.station])]
        self.station_owners = [None] * len(self.stations)
        self.shared_stations = "refill_points" in self.policy or self.policy.get("refill_layout") == "homes"
        self.drones = []
        homes = [(self.nav["start"][0] + dx, self.nav["start"][1] + dy, 0.3)
                 for dx, dy in formation_offsets(config)]
        if self.nav.get("start_layout") == "corners":
            homes = []
            for corner, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
                count = config["drone_count"] // 4 + (corner < config["drone_count"] % 4)
                columns = math.ceil(math.sqrt(count))
                for j in range(count):
                    homes.append((sx * (self.nav["corner_offset"] - j % columns * config["spacing"]),
                                  sy * (self.nav["corner_offset"] - j // columns * config["spacing"]), 0.3))
        if self.policy.get("refill_layout") == "homes":
            self.stations = [(*home[:2],self.station[2]) for home in homes]
            self.station_owners = [None]*len(self.stations)
        for i, home in enumerate(homes):
            lane = self.policy["altitude_lanes"][i % len(self.policy["altitude_lanes"])]
            angle = 2 * math.pi * i / config["drone_count"]
            holding = (self.station[0] + self.policy["holding_radius"] * math.cos(angle),
                       self.station[1] + self.policy["holding_radius"] * math.sin(angle), lane)
            if self.shared_stations:
                holding = (*home[:2], lane)
            self.drones.append(Drone(i + 1, home, home, lane, holding, self.policy["tank_capacity_l"]))
        self.fire_geometry = []
        self.observation = {}
        self.station_owner = None
        self.refill_queue = []
        self.next_allocation = 0.0
        self.decisions = []
        self.refill_events = []
        self.replans = 0
        self.wait_events = 0
        self.water_dropped_l = 0.0
        self.water_refilled_l = 0.0
        self.observed_edges = set()
        self.covered_edges = set()
        self._footprints = {}
        self._geometry_by_cell = None
        self.defense_policy = None
        self.distribution = [0.0]*SECTORS
        self.navigator = DroneNavigator(config)
        self.external_action = None
        self.next_policy_update = 0.0
        self.policy_logits = None
        self._numpy = None
        if config["fire"].get("model") == "lagrangian":
            import numpy as np
            self._numpy = np
            self._station_xy = np.asarray([point[:2] for point in self.stations])

    def footprint(self, center):
        key = tuple(center)
        if key not in self._footprints:
            if self._geometry_by_cell is None:
                self._geometry_by_cell = {tuple(item["cell"]):item for item in self.fire_geometry}
            size, radius = self.config["fire"]["cell_size"], self.policy["drop_radius"]
            nearby = [self._geometry_by_cell.get((x,y))
                      for x in range(math.ceil((center[0]-radius)/size),math.floor((center[0]+radius)/size)+1)
                      for y in range(math.ceil((center[1]-radius)/size),math.floor((center[1]+radius)/size)+1)]
            self._footprints[key] = [item for item in nearby if item is not None
                if (not self.policy.get("perimeter_only") or item["edges"])
                and math.dist(center, item["center"]) <= self.policy["drop_radius"] + 1e-9]
        return self._footprints[key]



    def allocate_defense(self, time):
        """Translate learned sector water fractions into safe individual targets."""
        fleet = fleet_summary(self.drones,self.policy["tank_capacity_l"])
        vector, sectors = observation_vector(self.config,self.observation,fleet,time)
        if self.external_action is not None:
            from drones.policy import action_fractions
            fractions = action_fractions(self.external_action,sectors)
        elif self.policy["allocation_model"] == "ppo":
            if self.defense_policy is None:
                path = Path(self.policy["ppo_checkpoint"])
                if not path.is_absolute(): path = Path(__file__).resolve().parents[1]/path
                if not path.is_file():
                    raise ValueError("PPO checkpoint absent; run .learning-venv/bin/python train_ppo.py --config configs/ppo-100-fast.json")
                self.defense_policy = PPOPolicy.load(path)
                self.defense_policy.validate_fire_model(self.config)
            from drones.policy import action_fractions
            if self.policy_logits is None or not self.policy.get("cache_policy_actions") or time >= self.next_policy_update-1e-9:
                self.policy_logits = self.defense_policy.predict(vector)
                self.next_policy_update = time+self.policy["decision_interval"]
            fractions = action_fractions(self.policy_logits,sectors)
        else:
            fractions = uniform_fractions(sectors)
        self.distribution = fractions
        by_cell = {tuple(item['cell']):(i,item) for i,sector in enumerate(sectors) for item in sector}
        goals = SegmentIndex(self.config["minimum_separation"])
        assigned, supplied = set(),[0.0]*SECTORS
        for d in self.drones:
            if d.mode in ("attack","drop") and d.target in by_cell:
                sector,_ = by_cell[d.target]
                assigned.add(d.target)
                supplied[sector] += d.water_l
                goals.update(d.number,d.goal,d.goal)
        candidates = sorted((d for d in self.drones if d.mode in ("attack","idle") and d.target not in assigned),
                            key=lambda d:(-d.water_l,d.number))
        total_water = sum(supplied)+sum(d.water_l for d in candidates)
        assignments = []
        remaining = [{tuple(item['cell']):item for item in sector if tuple(item['cell']) not in assigned} for sector in sectors]
        np = self._numpy
        sector_items = [list(items.values()) for items in remaining] if np is not None else None
        centers = [np.asarray([item['center'] for item in items]).reshape(-1,2) for items in sector_items] if np is not None else None
        available = [np.ones(len(items),dtype=bool) for items in sector_items] if np is not None else None
        for d in candidates:
            choice = None
            order = sorted(range(SECTORS),key=lambda i:fractions[i]*total_water-supplied[i],reverse=True)
            for sector in order:
                if np is not None:
                    distances = np.sum((centers[sector]-d.position[:2])**2,axis=1)
                    distances[~available[sector]] = np.inf
                    candidates_here = ((sector_items[sector][index],index) for index in np.argsort(distances)
                                       if math.isfinite(float(distances[index])))
                else:
                    candidates_here = ((item,None) for item in sorted(remaining[sector].values(),key=lambda cell:math.dist(d.position[:2],cell['center'])))
                for item,index in candidates_here:
                    goal = (*item['center'],d.lane)
                    if all(math.dist(goal,self.drones[number-1].goal)>=self.config["minimum_separation"]
                           for number in goals.query(goal,goal)):
                        choice = sector,item,goal
                        if np is not None: available[sector][index] = False
                        break
                if choice is not None: break
            if choice is None:
                d.target,d.goal,d.mode = None,d.holding,"idle"
                continue
            sector,item,goal = choice
            d.target,d.goal,d.mode = tuple(item['cell']),goal,"attack"
            remaining[sector].pop(d.target)
            goals.update(d.number,goal,goal)
            supplied[sector] += d.water_l
            self.replans += 1
            assignments.append({"drone":d.number,"cell":item['cell'],"goal":list(goal),"sector":sector})
        record = {"time":time,"observation":vector,"water_fractions":fractions,"assignments":assignments}
        if not self.policy.get("compact_log"):
            record.update(fire_geometry=self.fire_geometry,perimeter_edges=self.observation["perimeter_edges"])
        if self.policy.get("record_decisions",True):self.decisions.append(record)

    def arrived(self, drone, point):
        return math.dist(drone.position, point) < 1e-7

    def separated(self, a, b, c, d):
        return self.navigator.separated(a,b,c,d)

    def request_refill(self, drone, time):
        drone.target, drone.goal, drone.mode = None, drone.holding, "queued"
        if drone.number not in self.refill_queue:
            self.refill_queue.append(drone.number)
            self.refill_events.append({"time": time, "drone": drone.number, "event": "queued"})

    def next_direction(self, drone, dt):
        return self.navigator.next_direction(drone,dt)

    def fire_clear(self, start, end):
        self.navigator.fire_geometry=self.fire_geometry
        return self.navigator.fire_clear(start,end)

    def step(self, time, dt, observation):
        """Observe, assign, dispense/refill, then reserve this interval's moves.

        Returned drops occur at the current pose/time. The runner applies them
        to the fire *after* this decision; the next call sees their result.
        """
        self.observation = observation
        self.fire_geometry = observation["geometry"]
        self._geometry_by_cell = {tuple(item["cell"]):item for item in self.fire_geometry}
        self._footprints.clear()
        self.observed_edges.update(self.edge_key(edge) for edge in observation["perimeter_edges"])
        active = {tuple(cell) for cell in observation["active"]}
        if self.policy.get("perimeter_only"):
            active = {tuple(item["cell"]) for item in self.fire_geometry if item["edges"]}
        drops = []
        landing_start = self.config["duration"] - self.config["landing_duration"]
        return_start = landing_start - self.policy["return_duration"]
        need_allocation = time + 1e-9 >= self.next_allocation
        for drone in self.drones:
            if time < self.config["takeoff_duration"]:
                u = min(1.0, (time + dt) / self.config["takeoff_duration"])
                blend = u ** 3 * (10 + u * (-15 + 6 * u))
                drone.goal = (*drone.home[:2], 0.3 + (drone.lane - 0.3) * blend)
                continue
            if time >= return_start:
                drone.target = None
                if time < landing_start:
                    drone.mode = "returning"
                    drone.goal = ((*drone.position[:2], drone.lane)
                                  if abs(drone.position[2] - drone.lane) > 1e-7
                                  else (*drone.home[:2], drone.lane))
                else:
                    if math.dist(drone.position[:2], drone.home[:2]) > 1e-7:
                        raise ValueError("drone_%02d has not returned before landing (%.1f m remaining); increase return_duration"
                                         % (drone.number, math.dist(drone.position[:2], drone.home[:2])))
                    u = max(0.0, (self.config["duration"] - time - dt) / self.config["landing_duration"])
                    blend = u ** 3 * (10 + u * (-15 + 6 * u))
                    drone.mode, drone.goal = "landing", (*drone.home[:2], 0.3 + (drone.lane - 0.3) * blend)
                continue
            if drone.mode == "takeoff":
                drone.mode = "attack"
                need_allocation = True
            if drone.mode == "drop" and drone.drop_until is not None:
                if time < drone.drop_until-1e-9:
                    continue
                drone.drop_until = None
                drone.mode = "attack"
            if drone.mode in ("attack", "drop", "idle") and drone.water_l < 1e-8:
                self.request_refill(drone, time)
            if drone.mode in ("attack", "drop") and drone.target not in active:
                drone.mode, drone.target = "attack", None
                need_allocation = True
            if drone.mode == "attack" and drone.goal and drone.target in active and self.arrived(drone, drone.goal):
                drone.mode = "drop"
            if drone.mode == "drop":
                burst = self.policy["drop_duration"]
                amount = drone.water_l if dt else 0.0
                if amount > 0:
                    footprint = self.footprint(drone.position[:2])
                    if not self.policy.get("perimeter_only"):
                        self.covered_edges.update(self.edge_key(edge) for item in footprint for edge in item["edges"])
                    drops.append({"time": time, "drone": drone.number, "position": list(drone.position),
                                  "litres": amount, "radius": self.policy["drop_radius"]})
                    drone.water_l -= amount
                    self.water_dropped_l += amount
                    drone.drop_until = time+burst
            if drone.mode == "refilling":
                duration = self.policy['refill_duration']
                elapsed = max(0.0, time - drone.service_started)
                target_water = drone.service_water + (self.policy['tank_capacity_l'] - drone.service_water) * min(1.0, elapsed / duration)
                amount = max(0.0, target_water - drone.water_l)
                drone.water_l += amount
                self.water_refilled_l += amount
                if drone.water_l >= self.policy["tank_capacity_l"] - 1e-8 and elapsed + 1e-9 >= duration:
                    drone.mode, drone.goal = "depart_refill", drone.holding
                    self.refill_events.append({"time": time, "drone": drone.number, "event": "filled", "station": drone.station_index})
            station = self.stations[drone.station_index] if drone.station_index is not None else self.station
            if drone.mode == "to_refill" and self.shared_stations:
                # Fly to the station at cruising altitude before descending.
                drone.goal = ((*station[:2], drone.lane) if math.dist(drone.position[:2], station[:2]) > 1e-7 else station)
            if drone.mode == "to_refill" and self.arrived(drone, station):
                drone.mode = "refilling"
                drone.service_started, drone.service_water = time, drone.water_l
                self.refill_events.append({"time": time, "drone": drone.number, "event": "service_started", "station": drone.station_index})
            if drone.mode == "depart_refill":
                if self.shared_stations:
                    drone.goal = ((*drone.position[:2], drone.lane) if abs(drone.position[2] - drone.lane) > 1e-7 else drone.holding)
                    if drone.station_index is not None and math.dist(drone.position, station) > self.config["minimum_separation"] + 0.5:
                        self.station_owners[drone.station_index] = None
                        drone.station_index = None
                if self.station_owner == drone.number and math.dist(drone.position, self.station) > self.config["minimum_separation"] + 0.5:
                    self.station_owner = None
                if self.arrived(drone, drone.holding):
                    drone.mode = "attack"
                    need_allocation = True
        if self.config["takeoff_duration"] <= time < return_start:
            if self.shared_stations:
                for number in list(self.refill_queue):
                    drone = self.drones[number - 1]
                    free = [i for i, owner in enumerate(self.station_owners) if owner is None]
                    if not free:
                        break
                    if self._numpy is not None:
                        distance = self._numpy.sum((self._station_xy[free]-drone.position[:2])**2,axis=1)
                        index = free[int(self._numpy.argmin(distance))]
                    else:
                        index = min(free, key=lambda i: math.dist(drone.position, (*self.stations[i][:2], drone.lane)))
                    self.station_owners[index], drone.station_index = number, index
                    drone.mode, drone.goal = "to_refill", (*self.stations[index][:2], drone.lane)
                    self.refill_queue.remove(number)
                    self.refill_events.append({"time": time, "drone": number, "event": "reserved", "station": index})
            elif self.station_owner is None and self.refill_queue:
                first = self.drones[self.refill_queue[0] - 1]
                if self.arrived(first, first.holding):
                    self.refill_queue.pop(0)
                    self.station_owner = first.number
                    first.mode, first.goal = "to_refill", self.station
            if need_allocation:
                self.allocate_defense(time)
                self.next_allocation = time + self.policy["decision_interval"]
        if time >= return_start:
            self.station_owner, self.refill_queue = None, []
            self.station_owners = [None] * len(self.stations)
        self.navigator.fire_geometry=self.fire_geometry
        before, following, waits=self.navigator.reserve(self.drones,time,dt)
        self.wait_events += waits
        if waits and not self.shared_stations:
            self.next_allocation=min(self.next_allocation,time+dt)
        state = [{"drone": drone.number, "mode": drone.mode, "water_l": drone.water_l,
                  "lane": drone.lane, "position": list(before[i]),
                  "goal": list(drone.goal) if drone.goal else None,
                  "target_cell": list(drone.target) if drone.target else None, "blocked": drone.blocked,
                  "station": drone.station_index,
                  "direction": [(following[i][k]-before[i][k])/dt if dt else 0.0 for k in range(3)]}
                 for i, drone in enumerate(self.drones)]
        for drone, point in zip(self.drones, following):
            drone.position = point
        return {"drops": drops, "state": state}

    def metrics(self):
        covered = self.covered_edges & self.observed_edges
        return {"water_dropped_l": self.water_dropped_l, "water_refilled_l": self.water_refilled_l,
                "perimeter_edges_observed": len(self.observed_edges), "perimeter_edges_wetted": len(covered),
                "perimeter_coverage_fraction": len(covered) / len(self.observed_edges) if self.observed_edges else 0.0,
                "perimeter_wetted_m": len(covered) * self.config["fire"]["cell_size"],
                "refills_completed": sum(event["event"] == "filled" for event in self.refill_events),
                "replans": self.replans, "wait_events": self.wait_events,
                "policy": self.policy["allocation_model"]}
