"""Translate current fire snapshots into Gazebo visibility tracks."""
import math


def fire_data(config, fire, snapshots):
    if snapshots and "render_active" in snapshots[0]:
        tile = snapshots[0]["render_cell_size"]
        size = config["fire"]["cell_size"]
        x0,x1,y0,y1 = config["fire"]["domain"]
        cells = [(x,y) for x in range(math.floor(x0*size/tile),math.floor(x1*size/tile)+1)
                 for y in range(math.floor(y0*size/tile),math.floor(y1*size/tile)+1)]
        visible = [{tuple(cell) for cell in s["render_active"]} for s in snapshots]
        tracks = []
        for x,y in cells:
            rows = []
            for i,(snapshot,active) in enumerate(zip(snapshots,visible)):
                z = config["fire"]["height"]/2 if (x,y) in active else -50.0
                if not rows or rows[-1][3]!=z or i==len(snapshots)-1:
                    rows.append((snapshot["time"],(x+.5)*tile,(y+.5)*tile,z,0.0))
            tracks.append(rows)
        return {"cells":[list(cell) for cell in cells],"snapshots":snapshots,"obstacle_trajectories":tracks,
                "render_cell_size":tile,"simulation_cell_size":size,"simulation_cell_count":len(fire.cells),
                "snapshot_encoding":"fine_cell_deltas_and_render_tiles"}
    raise ValueError('Gazebo fire snapshots require render tiles')
