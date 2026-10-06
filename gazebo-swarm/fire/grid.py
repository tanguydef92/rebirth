"""Grid bookkeeping and geometry for the thermal particle model."""
import math

NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class FireGrid:
    def __init__(self, config):
        self.config = dict(config)
        x0, x1, y0, y1 = config["domain"]
        self.cells = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
        self.cell_set = set(self.cells)
        self.active = set()
        self.ignited = False
        self.recent_ignitions = set()
        self.extinguished_by_water = 0
        self.water_received = 0.0
        self.frontier = set()
        self.burned = set()

    def activate(self, cell):
        self.active.add(cell)
        self.burned.add(cell)
        for point in [cell] + [(cell[0]+dx, cell[1]+dy) for dx, dy in NEIGHBORS]:
            if point in self.active:
                if any((point[0]+dx, point[1]+dy) not in self.active for dx, dy in NEIGHBORS):
                    self.frontier.add(point)
                else:
                    self.frontier.discard(point)

    def deactivate(self, cell):
        self.active.discard(cell)
        self.frontier.discard(cell)
        self.frontier.update((cell[0]+dx, cell[1]+dy) for dx, dy in NEIGHBORS
                             if (cell[0]+dx, cell[1]+dy) in self.active)

    def footprint(self, point, radius):
        size = self.config["cell_size"]
        return [(x,y) for x in range(math.ceil((point[0]-radius)/size),math.floor((point[0]+radius)/size)+1)
                for y in range(math.ceil((point[1]-radius)/size),math.floor((point[1]+radius)/size)+1)
                if (x,y) in self.cell_set and math.dist((x*size,y*size),point[:2]) <= radius+1e-9]

    def ignite(self):
        cells = [tuple(cell) for cell in self.config['initial_cells']]
        if 'initial_rectangle' in self.config:
            x0, x1, y0, y1 = self.config['initial_rectangle']
            cells.extend((x,y) for x in range(x0,x1+1) for y in range(y0,y1+1))
        self.active = set(cells)
        self.burned.update(self.active)
        self.frontier = {cell for cell in self.active
                         if any((cell[0]+dx,cell[1]+dy) not in self.active for dx,dy in NEIGHBORS)}
        self.ignited = True
        self.recent_ignitions = set(self.active)
