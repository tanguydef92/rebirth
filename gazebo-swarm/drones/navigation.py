"""Bounded 3D steering and continuous collision reservations."""
import math
from drones.spatial import SegmentIndex
from drones.geometry import intersects_rectangle, segment_separation

class DroneNavigator:
    def __init__(self, config):
        self.config=config
        self.nav=config["navigation"]
        self.policy=config["suppression"]
        self.fire_geometry=[]

    def separated(self, a, b, c, d):
        limit = self.config["minimum_separation"] - 1e-9
        # A separating axis proves safety for the entire linear interval.
        for k in (2, 0, 1):
            start, end = a[k] - c[k], b[k] - d[k]
            if min(start, end) >= limit or max(start, end) <= -limit:
                return True
        return segment_separation(a, b, c, d) >= limit

    def next_direction(self, drone, dt):
        """Choose a bounded 3D movement from current geometry and drone state."""
        goal = drone.goal or drone.position
        delta = [goal[k] - drone.position[k] for k in range(3)]
        horizontal = math.hypot(*delta[:2])
        factor = min(1.0, self.nav["cruise_speed"] * dt / horizontal) if horizontal else 1.0
        move = [delta[0] * factor, delta[1] * factor,
                max(-self.nav.get("descent_speed", self.nav["vertical_speed"]) * dt,
                    min(self.nav["vertical_speed"] * dt, delta[2]))]
        length = math.sqrt(sum(v * v for v in move))
        if length > self.config["maximum_speed"] * dt and length:
            move = [v * self.config["maximum_speed"] * dt / length for v in move]
        return tuple(drone.position[k] + move[k] for k in range(3))

    def fire_clear(self, start, end):
        """Conservative continuous segment check against 3D fire boxes."""
        margin = self.policy.get("drone_radius", 0.5) + self.nav["clearance"]
        half = self.config["fire"]["cell_size"] / 2 + margin
        top = self.config["fire"]["height"] + margin
        if min(start[2], end[2]) > top:
            return True
        # Homes and stations lie outside the whole possible fire domain. Avoid
        # walking every burning cell for stationary refill and landing segments.
        x0, x1, y0, y1 = self.config["fire"]["domain"]
        size = self.config["fire"]["cell_size"]
        for axis, low, high in ((0, x0*size-half, x1*size+half), (1, y0*size-half, y1*size+half)):
            if max(start[axis], end[axis]) < low or min(start[axis], end[axis]) > high:
                return True
        for item in self.fire_geometry:
            cx, cy = item["center"]
            if min(start[2], end[2]) > top:
                continue
            # Split at the top plane to avoid rejecting safe high-altitude parts.
            a, b = start, end
            if (a[2] > top) != (b[2] > top):
                fraction = (top - a[2]) / (b[2] - a[2])
                crossing = tuple(a[k] + fraction * (b[k] - a[k]) for k in range(3))
                a, b = (crossing, b) if a[2] > top else (a, crossing)
            if intersects_rectangle(a, b, (cx-half, cx+half, cy-half, cy+half)):
                return False
        return True

    def reserve(self, drones, time, dt):
        waits=0
        before = [drone.position for drone in drones]
        following = list(before)
        spatial = SegmentIndex(self.config["minimum_separation"]) if len(drones)>=32 else None
        if spatial:
            for i,point in enumerate(before): spatial.update(i,point,point)
        def clear_peers(i,point):
            peers = spatial.query(before[i],point) if spatial else range(len(drones))
            return all(self.separated(before[i],point,before[j],following[j]) for j in peers if j!=i)
        for offset in range(len(drones)):
            i = (round(time / dt) + offset) % len(drones) if dt else offset
            drone = drones[i]
            candidate = self.next_direction(drone, dt)
            safe = self.fire_clear(before[i], candidate) and clear_peers(i,candidate)
            drone.blocked = not safe
            if safe:
                following[i] = candidate
            else:
                # Try lateral deviations at the assigned altitude before waiting.
                # This releases encounters between drones sharing the same lane.
                dx, dy = candidate[0] - before[i][0], candidate[1] - before[i][1]
                if math.hypot(dx, dy) < 1e-10:
                    heading = drone.home if drone.mode == "returning" else drone.holding
                    hx, hy = heading[0] - before[i][0], heading[1] - before[i][1]
                    length = math.hypot(hx, hy)
                    if length:
                        dx, dy = hx / length * self.nav["cruise_speed"] * dt, hy / length * self.nav["cruise_speed"] * dt
                    else:
                        dx, dy = self.nav["cruise_speed"] * dt, 0.0
                alternatives = [
                    (before[i][0] - side*dy, before[i][1] + side*dx, before[i][2])
                    for side in (1, -1)] + [
                    (before[i][0] + .5*dx - side*.866*dy,
                     before[i][1] + .5*dy + side*.866*dx, before[i][2])
                    for side in (1, -1)] + [(*before[i][:2], candidate[2])]
                for alternative in alternatives:
                    if any(abs(alternative[k]) > self.nav["world_bound"] for k in (0, 1)):
                        continue
                    if self.fire_clear(before[i], alternative) and clear_peers(i,alternative):
                        following[i] = alternative
                        break
                waits += 1
            if spatial:
                spatial.update(i,before[i],following[i])
        return before, following, waits
