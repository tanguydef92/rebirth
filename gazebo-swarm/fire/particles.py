"""Reduced thermal / Lagrangian fire inspired by Petersen et al. (2023).

The paper supplies the modelling concepts and latent-heat water law, but not a
complete implementation. These equations and demo coefficients are an explicit
reduced model, not a reproduction or calibration of the published simulator.
Terrain stores fuel mass and temperature. Weighted gas parcels carry sensible
energy and reacting volatiles; wind plus OU fluctuations transport them.
Conservative merging keeps at most one parcel per terrain cell at substep ends.
NumPy is imported only when this model is selected.
"""
import math
import copy

import numpy as np

from fire.grid import FireGrid

DEFAULT_PHYSICS = {
    "ambient_temperature_k": 293.15,
    "ignition_temperature_k": 600.0,
    "initial_temperature_k": 900.0,
    "fuel_load_kg_m2": 1.5,
    "inert_load_kg_m2": 1.0,
    "solid_specific_heat_j_kg_k": 1500.0,
    "gas_specific_heat_j_kg_k": 1000.0,
    "heat_of_combustion_j_kg": 16000000.0,
    "pyrolysis_rate_at_ignition_s": 0.005,
    "activation_energy_j_mol": 35000.0,
    "maximum_pyrolysis_rate_s": 0.02,
    "burnout_fraction": 0.05,
    "initial_reaction_fraction": 0.3,
    "gas_reaction_rate_s": 0.8,
    "initial_air_fuel_ratio": 8.0,
    "entrainment_rate_s": 0.08,
    "heat_exchange_rate_s": 2.0,
    "terrain_cooling_time_s": 180.0,
    "gas_cooling_time_s": 20.0,
    "parcel_lifetime_s": 60.0,
    "wind_m_s": [2.0, 0.0],
    "transport_factor": 0.5,
    "turbulence_rms_m_s": 0.7,
    "turbulence_time_s": 3.0,
    "maximum_substep_s": 0.25,
    "water_latent_heat_j_kg": 2260000.0,
    "water_density_kg_l": 1.0,
    "water_cooling_efficiency": 0.35,
}
REGION_PROPERTIES = {
    "fuel_load_kg_m2", "inert_load_kg_m2", "solid_specific_heat_j_kg_k"
}
PARCEL_FIELDS = ("x", "y", "vx", "vy", "mass", "energy", "volatile", "age")


def validate_particle_config(config):
    """Validate before allocating arrays or starting a long simulation."""
    physics = config.get("physics", {})
    if not isinstance(physics, dict) or set(physics) - set(DEFAULT_PHYSICS):
        raise ValueError("Unknown fire.physics fields; see fire.particles.DEFAULT_PHYSICS")
    parameters = {**DEFAULT_PHYSICS, **physics}
    nonnegative = {"fuel_load_kg_m2", "activation_energy_j_mol", "pyrolysis_rate_at_ignition_s",
                   "initial_reaction_fraction", "gas_reaction_rate_s", "initial_air_fuel_ratio",
                   "entrainment_rate_s", "turbulence_rms_m_s"}
    for key, value in parameters.items():
        if key == "wind_m_s":
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("wind_m_s needs [vx, vy] in m/s")
            values = value
        else:
            values = [value]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in values):
            raise ValueError("fire.physics." + key + " must be finite and numeric")
        if key != "wind_m_s" and (value < 0 if key in nonnegative else value <= 0):
            raise ValueError("Invalid fire.physics." + key)
    if not parameters["ambient_temperature_k"] < parameters["ignition_temperature_k"] <= parameters["initial_temperature_k"]:
        raise ValueError("Require ambient < ignition <= initial temperature")
    if not 0 < parameters["burnout_fraction"] < 1:
        raise ValueError("burnout_fraction must be between 0 and 1")
    if not 0 <= parameters["initial_reaction_fraction"] <= 1 or not 0 < parameters["water_cooling_efficiency"] <= 1:
        raise ValueError("Reaction fraction and water efficiency must be fractions")
    if parameters["maximum_substep_s"] > config["update_interval"]:
        raise ValueError("maximum_substep_s must not exceed update_interval")
    regions = config.get("terrain_regions", [])
    if not isinstance(regions, list):
        raise ValueError("terrain_regions must be a list")
    x0, x1, y0, y1 = config["domain"]
    for region in regions:
        if (not isinstance(region, dict) or "bounds" not in region
                or set(region) - REGION_PROPERTIES - {"bounds"} or len(region) < 2):
            raise ValueError("Terrain regions need bounds and at least one material property")
        bounds = region["bounds"]
        if (not isinstance(bounds, list) or len(bounds) != 4 or any(type(v) is not int for v in bounds)
                or not x0 <= bounds[0] <= bounds[1] <= x1 or not y0 <= bounds[2] <= bounds[3] <= y1):
            raise ValueError("Terrain region bounds must lie inside the fire domain")
        for key in REGION_PROPERTIES & set(region):
            value = region[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or (value < 0 if key == "fuel_load_kg_m2" else value <= 0)):
                raise ValueError("Invalid terrain property: " + key)
    return parameters


class ParticleFire(FireGrid):
    """Cell temperatures determine ignition; probabilities do not advance fire."""
    continuous = True

    def clone(self):
        """Copy evolving state while sharing immutable terrain and grid geometry."""
        result = copy.copy(self)
        result.config = copy.deepcopy(self.config)
        result.physics = copy.deepcopy(self.physics)
        for name in ("temperature", "fuel", "flames", "ever_burning"):
            setattr(result,name,getattr(self,name).copy())
        for name in ("active", "frontier", "burned", "recent_ignitions"):
            setattr(result,name,getattr(self,name).copy())
        result.parcels={key:value.copy() for key,value in self.parcels.items()}
        return result

    def __init__(self, config):
        self.physics = validate_particle_config(config)
        super().__init__(config)
        self.x0, x1, self.y0, y1 = config["domain"]
        self.nx, self.ny = x1 - self.x0 + 1, y1 - self.y0 + 1
        self.size = config["cell_size"]
        self.count = self.nx * self.ny
        self.ambient = self.physics["ambient_temperature_k"]
        self.ignition = self.physics["ignition_temperature_k"]
        self.temperature = np.full(self.count, self.ambient, dtype=np.float64)
        shape = (self.ny, self.nx)
        properties = {name: np.full(shape, self.physics[name], dtype=np.float64) for name in REGION_PROPERTIES}
        for region in config.get("terrain_regions", []):
            xa, xb, ya, yb = region["bounds"]
            for key in REGION_PROPERTIES & set(region):
                properties[key][ya-self.y0:yb-self.y0+1, xa-self.x0:xb-self.x0+1] = region[key]
        self.initial_fuel = properties["fuel_load_kg_m2"].ravel() * self.size**2
        self.fuel = self.initial_fuel.copy()
        self.inert = properties["inert_load_kg_m2"].ravel() * self.size**2
        self.specific_heat = properties["solid_specific_heat_j_kg_k"].ravel()
        self.flames = np.zeros(self.count, dtype=bool)
        self.ever_burning = np.zeros(self.count, dtype=bool)
        self.parcels = {field: np.empty(0, dtype=np.float64) for field in PARCEL_FIELDS}
        self.last_time = None
        self.integration_tick = 0
        self.fuel_consumed_kg = self.fuel_removed_kg = 0.0
        self.water_cooling_j = self.water_evaporated_l = 0.0
        self.ambient_loss_j = self.escaped_energy_j = self.ignition_energy_j = 0.0
        self.peak_parcels = 0

    def index(self, cell):
        return (cell[1] - self.y0) * self.nx + cell[0] - self.x0

    def cells_from_indices(self, indices):
        return {(int(i % self.nx) + self.x0, int(i // self.nx) + self.y0) for i in indices}

    def heat_capacity(self):
        return (self.fuel + self.inert) * self.specific_heat

    def combustible(self):
        return (self.initial_fuel > 0) & (self.fuel > self.initial_fuel * self.physics["burnout_fraction"])

    def _normal(self, keys, purpose):
        """Cell/substep-keyed draws, unaffected by the policy's parcel counts."""
        def uniform(salt):
            value = np.asarray(keys, dtype=np.uint64) * np.uint64(73856093)
            value ^= np.uint64((self.config["seed"] * 2654435761) & ((1 << 64) - 1))
            value ^= np.uint64((self.integration_tick * 83492791) & ((1 << 64) - 1))
            value ^= np.uint64(salt)
            value = (value ^ (value >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
            value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
            return ((value ^ (value >> np.uint64(31))) >> np.uint64(11)).astype(np.float64) / 9007199254740992
        u = np.maximum(uniform(purpose), np.finfo(float).tiny)
        return np.sqrt(-2 * np.log(u)) * np.cos(2 * math.pi * uniform(purpose + 1))

    def _emit(self, dt):
        indices = np.flatnonzero(self.flames & self.combustible())
        if not len(indices):
            return
        p = self.physics
        exponent = p["activation_energy_j_mol"] / 8.314462618 * (1/self.ignition - 1/self.temperature[indices])
        rate = np.minimum(p["maximum_pyrolysis_rate_s"], p["pyrolysis_rate_at_ignition_s"] * np.exp(np.clip(exponent, -60, 60)))
        released = self.fuel[indices] * (-np.expm1(-rate * dt))
        self.fuel[indices] -= released
        self.fuel_consumed_kg += float(released.sum())
        keep = released > 0
        indices, released = indices[keep], released[keep]
        if not len(indices):
            return
        sensible = released * self.specific_heat[indices] * (self.temperature[indices] - self.ambient)
        sigma = p["turbulence_rms_m_s"]
        emitted = {
            "x": (indices % self.nx + self.x0) * self.size,
            "y": (indices // self.nx + self.y0) * self.size,
            "vx": p["wind_m_s"][0] + sigma * self._normal(indices, 10),
            "vy": p["wind_m_s"][1] + sigma * self._normal(indices, 20),
            "mass": released * (1 + p["initial_air_fuel_ratio"]),
            "energy": sensible + released * p["heat_of_combustion_j_kg"] * p["initial_reaction_fraction"],
            "volatile": released * (1 - p["initial_reaction_fraction"]),
            "age": np.zeros(len(indices)),
        }
        self.parcels = {name: np.concatenate((self.parcels[name], emitted[name])) for name in PARCEL_FIELDS}

    def _discard(self, keep):
        p = self.parcels
        self.escaped_energy_j += float((p["energy"][~keep] + p["volatile"][~keep] * self.physics["heat_of_combustion_j_kg"]).sum())
        self.parcels = {name: values[keep] for name, values in p.items()}

    def _merge(self, indices):
        """Mix parcels within a fine cell, conserving mass, heat and momentum."""
        p = self.parcels
        mass = np.bincount(indices, weights=p["mass"], minlength=self.count)
        occupied = np.flatnonzero(mass > 0)
        merged = {"mass": mass[occupied]}
        for name in ("energy", "volatile"):
            merged[name] = np.bincount(indices, weights=p[name], minlength=self.count)[occupied]
        for name in ("x", "y", "vx", "vy", "age"):
            merged[name] = np.bincount(indices, weights=p[name] * p["mass"], minlength=self.count)[occupied] / merged["mass"]
        self.parcels = merged
        self.peak_parcels = max(self.peak_parcels, len(occupied))
        return occupied

    def _advance(self, dt):
        p = self.physics
        self.integration_tick += 1
        self._emit(dt)
        parcel = self.parcels
        if len(parcel["mass"]):
            # Emitted and advected parcels in the same cell share a keyed wind draw.
            ix = np.floor(parcel["x"] / self.size + .5).astype(np.int64) - self.x0
            iy = np.floor(parcel["y"] / self.size + .5).astype(np.int64) - self.y0
            keys = iy * self.nx + ix
            relaxation = math.exp(-dt / p["turbulence_time_s"])
            noise = p["turbulence_rms_m_s"] * math.sqrt(1 - relaxation**2)
            for name, wind, salt in (("vx", p["wind_m_s"][0], 30), ("vy", p["wind_m_s"][1], 40)):
                parcel[name] = wind + relaxation * (parcel[name] - wind) + noise * self._normal(keys, salt)
            # Substeps limit travel; exchange is at the destination fine cell.
            parcel["x"] += p["transport_factor"] * parcel["vx"] * dt
            parcel["y"] += p["transport_factor"] * parcel["vy"] * dt
            parcel["age"] += dt
            reacted = parcel["volatile"] * (-math.expm1(-p["gas_reaction_rate_s"] * dt))
            parcel["volatile"] -= reacted
            parcel["energy"] += reacted * p["heat_of_combustion_j_kg"]
            parcel["mass"] *= math.exp(p["entrainment_rate_s"] * dt)
            lost = parcel["energy"] * (-math.expm1(-dt / p["gas_cooling_time_s"]))
            self.ambient_loss_j += float(lost.sum())
            parcel["energy"] -= lost
            ix = np.floor(parcel["x"] / self.size + .5).astype(np.int64) - self.x0
            iy = np.floor(parcel["y"] / self.size + .5).astype(np.int64) - self.y0
            keep = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny) & (parcel["age"] < p["parcel_lifetime_s"])
            self._discard(keep)
            occupied = self._merge((iy[keep] * self.nx + ix[keep]))
            if len(occupied):
                parcel = self.parcels
                solid_capacity = self.heat_capacity()[occupied]
                gas_capacity = parcel["mass"] * p["gas_specific_heat_j_kg_k"]
                gas_temperature = self.ambient + parcel["energy"] / gas_capacity
                exchange = ((gas_temperature - self.temperature[occupied])
                            * solid_capacity * gas_capacity / (solid_capacity + gas_capacity)
                            * (-math.expm1(-p["heat_exchange_rate_s"] * dt)))
                self.temperature[occupied] += exchange / solid_capacity
                parcel["energy"] -= exchange
                # Cold, fully reacted parcels can no longer heat a cell to ignition.
                keep = (parcel["volatile"] > 1e-10) | (self.ambient + parcel["energy"] / gas_capacity >= self.ignition)
                self._discard(keep)
        capacity = self.heat_capacity()
        lost = capacity * (self.temperature - self.ambient) * (-math.expm1(-dt / p["terrain_cooling_time_s"]))
        self.temperature -= lost / capacity
        self.ambient_loss_j += float(lost.sum())
        self.flames = self.combustible() & (self.temperature >= self.ignition)
        newly_burning = self.flames & ~self.ever_burning
        self.burned.update(self.cells_from_indices(np.flatnonzero(newly_burning)))
        self.ever_burning |= self.flames

    def step(self, time):
        if not math.isfinite(time) or time < 0:
            raise ValueError("Fire time must be finite and nonnegative")
        if self.last_time is not None and time < self.last_time - 1e-9:
            raise ValueError("Thermal fire cannot run backwards")
        ignition_time = self.config["ignition_time"]
        if time < ignition_time - 1e-9:
            return self.active
        if not self.ignited:
            self.ignite()
            seeds = [self.index(cell) for cell in self.active if self.initial_fuel[self.index(cell)] > 0]
            self.active = self.cells_from_indices(seeds)
            self.frontier.clear()
            for cell in list(self.active):
                self.activate(cell)
            self.burned = set(self.active)
            self.temperature[seeds] = self.physics["initial_temperature_k"]
            self.flames[seeds] = True
            self.ever_burning[seeds] = True
            self.ignition_energy_j = float((self.heat_capacity() * (self.temperature - self.ambient)).sum())
            self.last_time = ignition_time
        previous = set(self.active)
        remaining = max(0.0, time - self.last_time)
        # A conservative transport bound prevents skipping fine cells on average.
        speed = (math.hypot(*self.physics["wind_m_s"]) + 6*self.physics["turbulence_rms_m_s"]) * self.physics["transport_factor"]
        quantum = min(self.physics["maximum_substep_s"], self.size / (2*speed) if speed else math.inf)
        while remaining > 1e-9:
            dt = min(remaining, quantum)
            self._advance(dt)
            remaining -= dt
        updated = self.cells_from_indices(np.flatnonzero(self.flames))
        for cell in previous - updated:
            self.deactivate(cell)
        for cell in updated - previous:
            self.activate(cell)
        self.recent_ignitions = updated - previous if time > ignition_time else set(updated)
        self.last_time = time
        return self.active


    def has_fire_potential(self):
        return bool(self.active) or bool(len(self.parcels["mass"]))

    def water_needed(self, cell):
        i = self.index(cell)
        capacity = (self.fuel[i] + self.inert[i]) * self.specific_heat[i]
        energy = capacity * max(0, self.temperature[i] - self.ignition + 1.0)
        p = self.physics
        return float(energy / (p["water_cooling_efficiency"] * p["water_density_kg_l"] * p["water_latent_heat_j_kg"]))

    def apply_water(self, point, litres, radius, allowed_cells=None):
        if isinstance(litres, bool) or not math.isfinite(litres) or litres < 0:
            raise ValueError("Water dose must be finite and nonnegative")
        cells = self.footprint(point, radius)
        if allowed_cells is not None:
            cells = [cell for cell in cells if cell in allowed_cells]
        if not cells or not litres:
            return {"cells": [], "extinguished": [], "retained_l": 0.0, "cooling_j": 0.0, "evaporated_l": 0.0}
        p = self.physics
        extinguished, cooling = [], 0.0
        for cell in cells:
            i = self.index(cell)
            capacity = (self.fuel[i] + self.inert[i]) * self.specific_heat[i]
            requested = litres / len(cells) * p["water_density_kg_l"] * p["water_latent_heat_j_kg"] * p["water_cooling_efficiency"]
            removed = min(requested, capacity * max(0, self.temperature[i] - self.ambient))
            self.temperature[i] -= removed / capacity
            cooling += removed
            if cell in self.active and self.temperature[i] < self.ignition:
                self.flames[i] = False
                self.deactivate(cell)
                extinguished.append(list(cell))
        evaporated = cooling / (p["water_latent_heat_j_kg"] * p["water_density_kg_l"])
        self.water_cooling_j += cooling
        self.water_evaporated_l += evaporated
        self.water_received += litres
        self.extinguished_by_water += len(extinguished)
        return {"cells": [list(cell) for cell in cells], "extinguished": extinguished,
                "retained_l": litres, "cooling_j": cooling, "evaporated_l": evaporated}

    def apply_firebreak(self, bounds):
        xa, xb, ya, yb = bounds
        if not self.x0 <= xa <= xb < self.x0+self.nx or not self.y0 <= ya <= yb < self.y0+self.ny:
            raise ValueError("Firebreak outside terrain")
        cells = [(x, y) for y in range(ya, yb+1) for x in range(xa, xb+1)]
        for cell in cells:
            i = self.index(cell)
            self.fuel_removed_kg += float(self.fuel[i])
            # Removing fuel also removes its sensible and chemical energy.
            self.escaped_energy_j += float(self.fuel[i] * (self.specific_heat[i] * (self.temperature[i]-self.ambient)
                                                          + self.physics["heat_of_combustion_j_kg"]))
            self.fuel[i] = 0
            self.flames[i] = False
            self.deactivate(cell)
        return cells

    def edge_spread_probability(self, source, target):
        """Current directional heat/ignition-margin indicator, not a spread rule."""
        j, i = self.index(target), self.index(source)
        if self.initial_fuel[j] <= 0 or self.fuel[j] <= self.initial_fuel[j]*self.physics["burnout_fraction"]:
            return 0.0
        p = self.physics
        dx, dy = target[0]-source[0], target[1]-source[1]
        wind = math.hypot(*p["wind_m_s"])
        isotropic = p["turbulence_rms_m_s"] + .1
        direction = max(0, isotropic + dx*p["wind_m_s"][0] + dy*p["wind_m_s"][1]) / (wind + 4*isotropic)
        exponent = p["activation_energy_j_mol"] / 8.314462618 * (1/self.ignition - 1/self.temperature[i])
        rate = min(p["maximum_pyrolysis_rate_s"], p["pyrolysis_rate_at_ignition_s"] * math.exp(max(-60,min(60,exponent))))
        available = self.fuel[i] * (-math.expm1(-rate*self.config["update_interval"])) * p["heat_of_combustion_j_kg"]
        needed = (self.fuel[j]+self.inert[j])*self.specific_heat[j]*max(1,self.ignition-self.temperature[j])
        return -math.expm1(-available*direction/needed)

    def observe(self, time, compact=False):
        """Batch thermal diagnostics; expose only current copied geometry."""
        cells=sorted(self.frontier if compact or self.config.get("front_geometry_only") else self.active)
        coordinates=np.asarray(cells,dtype=np.int64).reshape(-1,2)
        indices=(coordinates[:,1]-self.y0)*self.nx+coordinates[:,0]-self.x0
        offsets=np.array([1,-1,self.nx,-self.nx])
        neighbors=indices[:,None]+offsets
        valid=np.column_stack((coordinates[:,0]<self.x0+self.nx-1,coordinates[:,0]>self.x0,
                               coordinates[:,1]<self.y0+self.ny-1,coordinates[:,1]>self.y0))
        neighbors=np.clip(neighbors,0,self.count-1)
        exposed=~(self.flames[neighbors]&valid)
        p=self.physics
        exponent=p['activation_energy_j_mol']/8.314462618*(1/self.ignition-1/self.temperature[indices])
        rates=np.minimum(p['maximum_pyrolysis_rate_s'],p['pyrolysis_rate_at_ignition_s']*np.exp(np.clip(exponent,-60,60)))
        energy=self.fuel[indices]*(-np.expm1(-rates*self.config['update_interval']))*p['heat_of_combustion_j_kg']
        isotropic=p['turbulence_rms_m_s']+.1
        wind=math.hypot(*p['wind_m_s'])
        direction=np.maximum(0,isotropic+np.array([p['wind_m_s'][0],-p['wind_m_s'][0],p['wind_m_s'][1],-p['wind_m_s'][1]]))/(wind+4*isotropic)
        required=(self.fuel[neighbors]+self.inert[neighbors])*self.specific_heat[neighbors]*np.maximum(1,self.ignition-self.temperature[neighbors])
        risk=-np.expm1(-energy[:,None]*direction/required)
        risk*=valid&exposed&self.combustible()[neighbors]
        recent=np.zeros(self.count,dtype=bool)
        for cell in self.recent_ignitions:recent[self.index(cell)]=True
        growth=(recent[neighbors]&valid).sum(axis=1).tolist()
        capacities=(self.fuel[indices]+self.inert[indices])*self.specific_heat[indices]
        dose=(capacities*np.maximum(0,self.temperature[indices]-self.ignition+1)
              /(p['water_cooling_efficiency']*p['water_density_kg_l']*p['water_latent_heat_j_kg'])).tolist()
        fractions=np.divide(self.fuel[indices],self.initial_fuel[indices],out=np.zeros(len(indices)),where=self.initial_fuel[indices]>0).tolist()
        cooling=np.clip((p['initial_temperature_k']-self.temperature[indices])/(p['initial_temperature_k']-self.ambient),0,1).tolist()
        geometry,edges=[],[]
        directions=((1,0),(-1,0),(0,1),(0,-1))
        rows=zip(cells,exposed.tolist(),risk.sum(axis=1).tolist(),growth,
                 self.temperature[indices].tolist(),self.fuel[indices].tolist(),fractions,dose,cooling)
        half=self.size/2
        for (x,y),mask,spread,new,temp,fuel,fraction,water,cold in rows:
            local=[[x,y,dx,dy] for (dx,dy),show in zip(directions,mask) if show]
            edges.extend(local);cx,cy=x*self.size,y*self.size
            geometry.append({'cell':[x,y],'center':[cx,cy],'edges':local,
                             'polygon':[[cx-half,cy-half],[cx+half,cy-half],[cx+half,cy+half],[cx-half,cy+half]],
                             'spread_risk':spread,'recent_growth':new,'water_l':0.0,
                             'temperature_k':temp,'fuel_kg':fuel,'fuel_fraction':fraction,
                             'water_needed_l':water,'cooling_fraction':cold})
        return {'time':time,'active':[] if compact else [list(cell) for cell in sorted(self.active)],
                'geometry':geometry,'perimeter_edges':edges,'perimeter_m':len(edges)*self.size,
                'active_count':len(self.active),'burned_count':len(self.burned),
                'recent_ignitions':[list(cell) for cell in sorted(self.recent_ignitions&self.active)],
                'wet':[],'ignited':self.ignited,'model':'lagrangian',
                'particle_count':len(self.parcels['mass']),'fire_potential':self.has_fire_potential(),
                'physics_metrics':self.metrics()}

    def metrics(self):
        parcel = self.parcels
        stored = float((self.heat_capacity() * (self.temperature-self.ambient)).sum()
                       + parcel["energy"].sum() + parcel["volatile"].sum()*self.physics["heat_of_combustion_j_kg"]
                       + self.fuel.sum()*self.physics["heat_of_combustion_j_kg"])
        initial = float(self.initial_fuel.sum()*self.physics["heat_of_combustion_j_kg"] + self.ignition_energy_j)
        balance = initial-stored-self.ambient_loss_j-self.escaped_energy_j-self.water_cooling_j
        return {"fuel_consumed_kg":self.fuel_consumed_kg, "fuel_removed_kg":self.fuel_removed_kg,
                "remaining_fuel_kg":float(self.fuel.sum()), "water_cooling_j":self.water_cooling_j,
                "water_evaporated_l":self.water_evaporated_l, "particle_count":len(parcel["mass"]),
                "peak_particle_count":self.peak_parcels, "maximum_temperature_k":float(self.temperature.max()),
                "energy_balance_error_j":balance, "energy_balance_relative_error":abs(balance)/max(1,initial)}
