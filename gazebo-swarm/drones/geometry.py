"""Geometry used by current drone control and continuous trajectory checks."""
import math


def formation_offsets(config):
    columns = 2
    rows = math.ceil(config["drone_count"] / columns)
    return [((i % columns - 0.5) * config["spacing"],
             (i // columns - (rows - 1) / 2) * config["spacing"])
            for i in range(config["drone_count"])]


def intersects_rectangle(a, b, rectangle):
    """Closed segment / closed axis-aligned rectangle (continuous, not sampled)."""
    lower, upper = 0.0, 1.0
    for k, low, high in ((0, rectangle[0], rectangle[1]), (1, rectangle[2], rectangle[3])):
        delta = b[k] - a[k]
        if abs(delta) < 1e-12:
            if a[k] < low or a[k] > high:
                return False
        else:
            enter, leave = (low - a[k]) / delta, (high - a[k]) / delta
            if enter > leave:
                enter, leave = leave, enter
            lower, upper = max(lower, enter), min(upper, leave)
            if lower > upper:
                return False
    return True


def segment_separation(a, b, c, d):
    """Closest distance of two simultaneous linearly interpolated movements."""
    x, y = a[0] - c[0], a[1] - c[1]
    z = a[2] - c[2] if len(a) == 3 else 0.0
    vx, vy = b[0] - d[0] - x, b[1] - d[1] - y
    vz = b[2] - d[2] - z if len(a) == 3 else 0.0
    norm = vx*vx + vy*vy + vz*vz
    alpha = max(0.0, min(1.0, -(x*vx + y*vy + z*vz) / norm)) if norm else 0.0
    return math.sqrt((x + alpha*vx)**2 + (y + alpha*vy)**2 + (z + alpha*vz)**2)
