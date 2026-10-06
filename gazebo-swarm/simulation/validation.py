"""Independent checks for full-precision motion and water-drop records."""
import numpy as np

from drones.geometry import intersects_rectangle


def validate_fire_clearance(config, positions):
    """Certify clearance from every possible fire cell, including between ticks.

    Flight lanes are above the fire and low homes/stations are outside its
    domain. This stronger check needs no stored history of burning cells.
    """
    positions = np.asarray(positions)
    margin = config['navigation']['clearance'] + config['suppression'].get('drone_radius', .5)
    half = config['fire']['cell_size'] / 2 + margin
    top = config['fire']['height'] + margin
    size = config['fire']['cell_size']
    x0, x1, y0, y1 = config['fire']['domain']
    rectangle = (x0 * size - half, x1 * size + half,
                 y0 * size - half, y1 * size + half)
    before, after = positions[:-1], positions[1:]
    low = np.minimum(before, after)
    high = np.maximum(before, after)
    candidates = ((low[:, :, 2] <= top) &
                  (high[:, :, 0] >= rectangle[0]) & (low[:, :, 0] <= rectangle[1]) &
                  (high[:, :, 1] >= rectangle[2]) & (low[:, :, 1] <= rectangle[3]))
    for tick, drone in np.argwhere(candidates):
        a, b = before[tick, drone], after[tick, drone]
        if (a[2] > top) != (b[2] > top):
            crossing = a + (top - a[2]) / (b[2] - a[2]) * (b - a)
            a, b = (crossing, b) if a[2] > top else (a, crossing)
        if intersects_rectangle(a, b, rectangle):
            raise ValueError('Drone %d entered the possible fire volume at tick %d' % (drone + 1, tick))
    return {'continuous_fire_clearance_validated': True,
            'fire_clearance_scope': 'entire_possible_fire_domain'}


def validate_water_drops(config, drops, metrics):
    capacity = config['suppression']['tank_capacity_l']
    for drop in drops:
        if not 0 < drop['litres'] <= capacity + 1e-8:
            raise ValueError('Water dose exceeds drone capacity')
        if config['suppression'].get('perimeter_only'):
            perimeter = {tuple(edge[:2]) for edge in drop['perimeter_edges']}
            if not drop['cells'] or any(tuple(cell) not in perimeter for cell in drop['cells']):
                raise ValueError('Water was applied outside the observed current perimeter')
    total = sum(drop['litres'] for drop in drops)
    if abs(total - metrics['water_dropped_l']) > 1e-6:
        raise ValueError('Water-drop records disagree with the tank budget')
    expected = config['drone_count'] * capacity + metrics['water_refilled_l'] - total
    if abs(expected - metrics['remaining_tank_water_l']) > 1e-6:
        raise ValueError('Final water budget failed')
    return {'water_budget_validated': True, 'water_drop_count': len(drops),
            'perimeter_drop_records_validated': bool(config['suppression'].get('perimeter_only'))}
