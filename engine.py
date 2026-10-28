"""NaviPath prototype engine: land-masked ocean graph + A* (stage 1) + ACO (stage 2).

Design notes
------------
* The graph is a regular lat/lon lattice. A node exists for every lattice point that is
  water in the 1 km GLOBE land mask (global-land-mask package).
* Land/water is tested on EDGES, not only on nodes: every candidate edge is sampled every
  ~1 km and rejected if a single sample is land. This is what stops routes from cutting
  across headlands even when a coarse lattice has both end nodes in the water.
* Raising the node density (smaller `res`) lets the lattice follow narrow passages and
  coastlines more closely. Hand-placed "highway" corridors (the stand-in for TSS lanes in
  the paper) cover canals and straits that no lattice can resolve.
* Costs follow the paper: C(e) = w1*dist + w2*weather + w3*risk + w4*fuel, plus an optional
  coastal-margin term (prototype extra). Stage 1 (A*) can use the paper's simple cost
  (distance x (1 + risk_weight)); stage 2 (ACO) always optimises the full composite cost.
"""
from __future__ import annotations

import heapq
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
from global_land_mask import globe
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

R_EARTH = 6371.0088
LAT_MIN, LAT_MAX = -70.0, 75.0
SAMPLE_KM = 1.0
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


# ----------------------------------------------------------------------------- geometry
def hav_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def wrap_lon(lon):
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def lerp_segment(p, q, step_km=SAMPLE_KM):
    """Sample points along the straight (lat/lon-linear) segment p->q, antimeridian safe."""
    d = float(hav_km(p[0], p[1], q[0], q[1]))
    n = max(2, int(d / step_km) + 2)
    t = np.linspace(0.0, 1.0, n)
    dl = ((q[1] - p[1] + 180.0) % 360.0) - 180.0
    return p[0] + (q[0] - p[0]) * t, wrap_lon(p[1] + dl * t), d


def segment_is_water(p, q, step_km=SAMPLE_KM):
    la, lo, _ = lerp_segment(p, q, step_km)
    return not globe.is_land(la, lo).any()


# ----------------------------------------------------------------------------- parameters
@dataclass
class CostParams:
    w_dist: float = 0.4
    w_wx: float = 0.3
    w_risk: float = 0.2
    w_fuel: float = 0.1
    w_coast: float = 0.1          # prototype extra: keeps routes off the coast
    risk_pref: float = 1.0        # 1 = OPTIMAL, 3 = SAFER_OPTIMAL (risk counts more)
    sea_k: float = 1.0            # fuel penalty growth in rough seas
    coast_buffer_km: float = 30.0

    @property
    def min_mult(self):
        return self.w_dist + self.w_fuel


@dataclass
class AcoParams:
    ants: int = 30
    iters: int = 60
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.10
    band_km: float = 600.0
    seg_km: float = 800.0         # ACO window length along the A* path
    elitist: bool = True          # A* path kept as incumbent (never worse than A*)
    warm_start: bool = True       # pheromone initialised along the A* path
    seed: int = 7
    best_only: bool = True        # only the iteration-best ant deposits pheromone (more exploration)
    guidance: str = "bound"       # ant heuristic info: 'bound' = straight-line lower bound, 'cost_to_go' = exact in-band
    seed_boost: float = 1.0       # warm-start pheromone added along the A* path


# ----------------------------------------------------------------------------- the graph
class OceanGrid:
    def __init__(self, res: float, gateways=(), log=None):
        self.res = float(res)
        self.H = int(round((LAT_MAX - LAT_MIN) / self.res)) + 1
        self.W = int(round(360.0 / self.res))
        self.lat = LAT_MIN + self.res * np.arange(self.H)
        self.lon = -180.0 + self.res * np.arange(self.W)
        self.cell_km = self.res * 111.195
        self.warnings: list[str] = []
        self._log = log or (lambda s: None)
        self._load_or_build_masks()
        self._row_lengths()
        self._build_gateways(gateways)
        self._components()

    # ---- masks ---------------------------------------------------------------
    def _cache_path(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        return os.path.join(CACHE_DIR, f"grid_{self.res:.3f}_v2.npz")

    def _load_or_build_masks(self):
        cp = self._cache_path()
        if os.path.exists(cp):
            z = np.load(cp)
            self.water, self.E, self.N, self.NE, self.SE = z["water"], z["E"], z["N"], z["NE"], z["SE"]
            self.dist_land_km = z["dist_land_km"]
            self.near_i, self.near_j = z["near_i"], z["near_j"]
            return
        self._log(f"Building land/water graph at {self.res} deg (one-off, cached)...")
        H, W, res = self.H, self.W, self.res
        water = np.zeros((H, W), bool)
        E = np.zeros((H, W), bool)
        N = np.zeros((H, W), bool)
        NE = np.zeros((H, W), bool)
        SE = np.zeros((H, W), bool)

        def edge_valid(la_a, lo_a, la_b, lo_b, ns):
            t = np.linspace(0.0, 1.0, ns)
            la = la_a[..., None] + (la_b - la_a)[..., None] * t
            lo = lo_a[..., None] + (lo_b - lo_a)[..., None] * t
            return ~globe.is_land(la, wrap_lon(lo)).any(axis=-1)

        ns_s = int(math.ceil(self.cell_km / SAMPLE_KM)) + 1
        ns_d = int(math.ceil(self.cell_km * 1.4143 / SAMPLE_KM)) + 1
        chunk = max(1, int(4_000_000 / (W * ns_d)))
        for i0 in range(0, H, chunk):
            i1 = min(H, i0 + chunk)
            la = np.repeat(self.lat[i0:i1, None], W, axis=1)
            lo = np.repeat(self.lon[None, :], i1 - i0, axis=0)
            water[i0:i1] = ~globe.is_land(la, lo)
            E[i0:i1] = edge_valid(la, lo, la, lo + res, ns_s)
            N[i0:i1] = edge_valid(la, lo, la + res, lo, ns_s)
            NE[i0:i1] = edge_valid(la, lo, la + res, lo + res, ns_d)
            SE[i0:i1] = edge_valid(la, lo, la - res, lo + res, ns_d)
        N[-1] = False
        NE[-1] = False
        SE[0] = False
        # distance to land (cells -> km); wrap in longitude by padding
        pad = min(W // 2, int(200 / self.cell_km) + 4)
        wp = np.pad(water, ((0, 0), (pad, pad)), mode="wrap")
        d, idx = ndimage.distance_transform_edt(wp, return_indices=True)
        d = d[:, pad:-pad]
        self.dist_land_km = (d * self.cell_km).astype(np.float32)
        # nearest WATER cell for every cell (used to read cost fields next to the coast)
        wsrc = np.pad(~water, ((0, 0), (pad, pad)), mode="wrap")  # zero where water -> nearest water
        _, idw = ndimage.distance_transform_edt(wsrc, return_indices=True)
        ni = idw[0][:, pad:-pad]
        nj = (idw[1][:, pad:-pad] - pad) % W
        self.near_i, self.near_j = ni.astype(np.int32), nj.astype(np.int32)
        self.water, self.E, self.N, self.NE, self.SE = water, E, N, NE, SE
        np.savez_compressed(cp, water=water, E=E, N=N, NE=NE, SE=SE, dist_land_km=self.dist_land_km,
                            near_i=self.near_i, near_j=self.near_j)

    def _row_lengths(self):
        lat, res = self.lat, self.res
        self.lenE = hav_km(lat, 0.0, lat, res).tolist()
        self.lenN = float(hav_km(0.0, 0.0, res, 0.0))
        self.lenNE = hav_km(lat, 0.0, lat + res, res).tolist()
        self.lenSE = hav_km(lat, 0.0, lat - res, res).tolist()
        self.Ev = bytearray(self.E.ravel().astype(np.uint8).tobytes())
        self.Nv = bytearray(self.N.ravel().astype(np.uint8).tobytes())
        self.NEv = bytearray(self.NE.ravel().astype(np.uint8).tobytes())
        self.SEv = bytearray(self.SE.ravel().astype(np.uint8).tobytes())
        self.latr = np.radians(self.lat).tolist()
        self.lonr = np.radians(self.lon).tolist()

    # ---- highway corridors -------------------------------------------------------
    def _build_gateways(self, gateways):
        self.base = self.H * self.W
        self.xlat: list[float] = []
        self.xlon: list[float] = []
        self.x_adj: dict[int, list[tuple[int, float, bool]]] = {}
        self.gateway_lines: list[dict] = []

        def add_edge(a, b, km, trusted):
            self.x_adj.setdefault(a, []).append((b, km, trusted))
            self.x_adj.setdefault(b, []).append((a, km, trusted))

        for g in gateways:
            ids = []
            for la, lo in g["nodes"]:
                ids.append(self.base + len(self.xlat))
                self.xlat.append(float(la))
                self.xlon.append(float(lo))
            canal = g.get("canal")
            used = set()
            for k in range(len(ids) - 1):
                p, q = g["nodes"][k], g["nodes"][k + 1]
                trusted = bool(canal and canal[0] <= k < canal[1])
                if not trusted and not segment_is_water(p, q):
                    self.warnings.append(f"{g['name']}: segment {k}->{k + 1} touches land in the mask and was dropped")
                    continue
                add_edge(ids[k], ids[k + 1], float(hav_km(p[0], p[1], q[0], q[1])), trusted)
                used.update((ids[k], ids[k + 1]))
            for nid in ids:
                self._link_to_grid(nid)
            self.gateway_lines.append(dict(name=g["name"], nodes=g["nodes"], canal=canal))

    def _link_to_grid(self, nid, radius_km=90.0, max_links=3):
        la, lo = self.xlat[nid - self.base], self.xlon[nid - self.base]
        r = int(math.ceil(radius_km / self.cell_km)) + 1
        i0 = int(round((la - LAT_MIN) / self.res))
        j0 = int(round((lo + 180.0) / self.res))
        cands = []
        for i in range(max(0, i0 - r), min(self.H, i0 + r + 1)):
            for j in range(j0 - r, j0 + r + 1):
                jj = j % self.W
                if not self.water[i, jj]:
                    continue
                d = float(hav_km(la, lo, self.lat[i], self.lon[jj]))
                if d <= radius_km:
                    cands.append((d, i * self.W + jj, self.lat[i], self.lon[jj]))
        cands.sort()
        links = 0
        for d, gid, glat, glon in cands:
            if segment_is_water((la, lo), (glat, glon)):
                self.x_adj.setdefault(nid, []).append((gid, d, False))
                self.x_adj.setdefault(gid, []).append((nid, d, False))
                links += 1
                if links >= max_links:
                    break

    # ---- connectivity ----------------------------------------------------------------
    def _components(self):
        H, W = self.H, self.W
        idx = np.arange(H * W).reshape(H, W)
        rows, cols = [], []

        def add(a, b):
            rows.append(a.ravel())
            cols.append(b.ravel())

        add(idx[self.E], np.roll(idx, -1, axis=1)[self.E])
        add(idx[:-1][self.N[:-1]], idx[1:][self.N[:-1]])
        add(idx[:-1][self.NE[:-1]], np.roll(idx[1:], -1, axis=1)[self.NE[:-1]])
        add(idx[1:][self.SE[1:]], np.roll(idx[:-1], -1, axis=1)[self.SE[1:]])
        xr, xc = [], []
        for a, lst in self.x_adj.items():
            for b, _, _ in lst:
                xr.append(a)
                xc.append(b)
        if xr:
            rows.append(np.array(xr))
            cols.append(np.array(xc))
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        ntot = self.base + len(self.xlat)
        g = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(ntot, ntot)).tocsr()
        _, labels = connected_components(g, directed=False)
        wflat = self.water.ravel()
        counts = np.bincount(labels[: self.base][wflat])
        self.main_label = int(np.argmax(counts))
        self.labels = labels
        self.main_mask = (labels[: self.base] == self.main_label) & wflat

    # ---- helpers ---------------------------------------------------------------------
    def node_latlon(self, n):
        if n < self.base:
            i, j = divmod(n, self.W)
            return float(self.lat[i]), float(self.lon[j])
        return self.xlat[n - self.base], self.xlon[n - self.base]

    def edge_trusted(self, a, b):
        for m, _, t in self.x_adj.get(a, ()):
            if m == b:
                return t
        return False

    def snap(self, lat, lon, max_km=400.0):
        r = int(math.ceil(max_km / self.cell_km)) + 1
        i0 = int(round((lat - LAT_MIN) / self.res))
        j0 = int(round((lon + 180.0) / self.res))
        ii = np.arange(max(0, i0 - r), min(self.H, i0 + r + 1))
        jj = (np.arange(j0 - r, j0 + r + 1)) % self.W
        gid = ii[:, None] * self.W + jj[None, :]
        ok = self.main_mask[gid]
        d = hav_km(lat, lon, self.lat[ii][:, None], self.lon[jj][None, :])
        d = np.where(ok, d, np.inf)
        k = np.unravel_index(np.argmin(d), d.shape)
        if not np.isfinite(d[k]) or d[k] > max_km:
            raise ValueError("No connected water node near this position")
        g = int(gid[k])
        la, lo = self.node_latlon(g)
        return g, la, lo, float(d[k])

    def neighbors(self, n):
        """Yield (neighbor_id, km) for node n."""
        out = []
        W, H = self.W, self.H
        if n < self.base:
            i, j = divmod(n, W)
            jp = j + 1 if j + 1 < W else 0
            jm = j - 1 if j > 0 else W - 1
            if self.Ev[n]:
                out.append((i * W + jp, self.lenE[i]))
            m = i * W + jm
            if self.Ev[m]:
                out.append((m, self.lenE[i]))
            if i < H - 1:
                if self.Nv[n]:
                    out.append((n + W, self.lenN))
                if self.NEv[n]:
                    out.append(((i + 1) * W + jp, self.lenNE[i]))
                m = (i + 1) * W + jm
                if self.SEv[m]:
                    out.append((m, self.lenSE[i + 1]))
            if i > 0:
                m = n - W
                if self.Nv[m]:
                    out.append((m, self.lenN))
                m = (i - 1) * W + jm
                if self.NEv[m]:
                    out.append((m, self.lenNE[i - 1]))
                if self.SEv[n]:
                    out.append(((i - 1) * W + jp, self.lenSE[i]))
        xa = self.x_adj.get(n)
        if xa:
            out.extend((m, km) for m, km, _ in xa)
        return out

    def stats(self):
        nodes = int(self.main_mask.sum())
        edges = int(self.E.sum() + self.N.sum() + self.NE.sum() + self.SE.sum())
        return dict(res=self.res, grid=f"{self.H} x {self.W}", water_nodes=int(self.water.sum()),
                    connected_nodes=nodes, edges=edges, corridors=len(self.gateway_lines),
                    corridor_nodes=len(self.xlat), warnings=list(self.warnings))


_GRIDS: dict[float, OceanGrid] = {}


def get_grid(res, gateways, log=None):
    key = round(float(res), 4)
    if key not in _GRIDS:
        _GRIDS[key] = OceanGrid(res, gateways, log)
    return _GRIDS[key]


# ----------------------------------------------------------------------------- scenario fields
@dataclass
class Fields:
    wx: np.ndarray       # weather severity 0..1
    rk: np.ndarray       # risk level 0 / 0.5 / 1
    blocked: np.ndarray  # restricted (impassable)


def _in_box(lat2, lon2, z):
    lat_ok = (lat2 >= z["lat_min"]) & (lat2 <= z["lat_max"])
    lo0, lo1 = z["lon_min"], z["lon_max"]
    if lo0 <= lo1:
        lon_ok = (lon2 >= lo0) & (lon2 <= lo1)
    else:  # crosses the antimeridian
        lon_ok = (lon2 >= lo0) | (lon2 <= lo1)
    return lat_ok & lon_ok


def rasterize(grid: OceanGrid, risk_zones, storms, restricted) -> Fields:
    lat2 = grid.lat[:, None] * np.ones((1, grid.W))
    lon2 = np.ones((grid.H, 1)) * grid.lon[None, :]
    wx = np.zeros((grid.H, grid.W), np.float32)
    rk = np.zeros_like(wx)
    bl = np.zeros((grid.H, grid.W), bool)
    for z in risk_zones:
        rk[_in_box(lat2, lon2, z)] = np.maximum(rk[_in_box(lat2, lon2, z)], float(z["level"]))
    for s in storms:
        d = hav_km(s["lat"], s["lon"], lat2, lon2)
        v = float(s["severity"]) * np.clip(1.0 - (d / float(s["radius_km"])) ** 2, 0.0, 1.0)
        wx = np.maximum(wx, v.astype(np.float32))
    for z in restricted:
        bl |= _in_box(lat2, lon2, z)
    return Fields(wx, rk, bl)


def composite_field(grid, f: Fields, cp: CostParams):
    coast = np.clip(1.0 - grid.dist_land_km / cp.coast_buffer_km, 0.0, 1.0)
    M = (cp.w_dist + cp.w_wx * f.wx + cp.w_risk * cp.risk_pref * f.rk
         + cp.w_fuel * (1.0 + cp.sea_k * f.wx) + cp.w_coast * coast)
    M = M.astype(np.float64)
    M[~grid.water | f.blocked] = np.inf
    return M


def astar_field(grid, f: Fields, safer: bool):
    """Paper stage-1 cost: distance x (1 + risk_weight); risk_weight = 0 for OPTIMAL."""
    M = np.ones((grid.H, grid.W))
    if safer:
        M = M + f.rk
    M[~grid.water | f.blocked] = np.inf
    return M


def flat_with_extras(grid, M, base_value):
    """Flatten a (H,W) multiplier field and append values for corridor nodes."""
    xs = []
    for la, lo in zip(grid.xlat, grid.xlon):
        i = min(max(int(round((la - LAT_MIN) / grid.res)), 0), grid.H - 1)
        j = int(round((lo + 180.0) / grid.res)) % grid.W
        v = M[i, j]
        xs.append(float(v) if np.isfinite(v) else base_value)
    return M.ravel().tolist() + xs


# ----------------------------------------------------------------------------- A*
def astar(grid: OceanGrid, Mall, src, dst, hscale, max_expansions=6_000_000):
    t0 = time.perf_counter()
    base, W = grid.base, grid.W
    gla, glo = grid.node_latlon(dst)
    gla, glo = math.radians(gla), math.radians(glo)
    cgla = math.cos(gla)
    latr, lonr = grid.latr, grid.lonr
    xlatr = [math.radians(v) for v in grid.xlat]
    xlonr = [math.radians(v) for v in grid.xlon]
    hcache = {}
    sin, cos, asin, sqrt = math.sin, math.cos, math.asin, math.sqrt

    def h(n):
        v = hcache.get(n)
        if v is None:
            if n < base:
                i, j = divmod(n, W)
                la, lo = latr[i], lonr[j]
            else:
                la, lo = xlatr[n - base], xlonr[n - base]
            s = sin((gla - la) / 2) ** 2 + cos(la) * cgla * sin((glo - lo) / 2) ** 2
            v = hscale * 2 * R_EARTH * asin(sqrt(s))
            hcache[n] = v
        return v

    g = {src: 0.0}
    came = {}
    heap = [(h(src), src)]
    closed = set()
    expanded = 0
    neighbors = grid.neighbors
    while heap:
        f, n = heapq.heappop(heap)
        if n in closed:
            continue
        if n == dst:
            break
        closed.add(n)
        expanded += 1
        if expanded > max_expansions:
            return None
        gn = g[n]
        mn = Mall[n]
        for m, km in neighbors(n):
            mm = Mall[m]
            if mm == math.inf or m in closed:
                continue
            ng = gn + km * 0.5 * (mn + mm)
            if ng < g.get(m, math.inf):
                g[m] = ng
                came[m] = n
                heapq.heappush(heap, (ng + h(m), m))
    else:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(came[path[-1]])
    path.reverse()
    return dict(path=path, cost=g[dst], expanded=expanded, seconds=time.perf_counter() - t0, hscale=hscale)


# ----------------------------------------------------------------------------- ACO (stage 2)
def _band_nodes(grid, path, buffer_km):
    base, W, H = grid.base, grid.W, grid.H
    ij = [divmod(n, W) for n in path if n < base]
    if not ij:
        return set(), set(n for n in path)
    ii = [a for a, _ in ij]
    jj = []
    prev = None
    for _, j in ij:  # unwrap longitude index
        if prev is not None:
            while j - prev > W / 2:
                j -= W
            while prev - j > W / 2:
                j += W
        jj.append(j)
        prev = j
    b = int(math.ceil(buffer_km / grid.cell_km))
    i0, i1 = max(0, min(ii) - b), min(H - 1, max(ii) + b)
    j0, j1 = min(jj) - b, max(jj) + b
    mask = np.zeros((i1 - i0 + 1, j1 - j0 + 1), bool)
    for a, c in zip(ii, jj):
        mask[a - i0, c - j0] = True
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=b)
    ai, aj = np.nonzero(mask)
    gids = (ai + i0) * W + ((aj + j0) % W)
    gids = np.unique(gids)
    gids = gids[grid.main_mask[gids]]
    return set(int(x) for x in gids), set(n for n in path if n >= base)


def _aco_window(grid: OceanGrid, Mall, path, ap: AcoParams, hscale, rng):
    t0 = time.perf_counter()
    band, extras = _band_nodes(grid, path, ap.band_km)
    nodes = sorted(band | extras | set(path))
    nodes = [n for n in nodes if Mall[n] != math.inf]
    lid = {n: k for k, n in enumerate(nodes)}
    n_loc = len(nodes)
    # heuristic potentials toward the destination
    dst_g = path[-1]
    gla, glo = grid.node_latlon(dst_g)
    pot = np.array([hscale * float(hav_km(*grid.node_latlon(n), gla, glo)) for n in nodes])
    nbr_lists, cost_lists = [], []
    K = 1
    for n in nodes:
        nl, cl = [], []
        mn = Mall[n]
        for m, km in grid.neighbors(n):
            k = lid.get(m)
            if k is None:
                continue
            nl.append(k)
            cl.append(km * 0.5 * (mn + Mall[m]))
        nbr_lists.append(nl)
        cost_lists.append(cl)
        K = max(K, len(nl))
    nbr = np.full((n_loc, K), -1, np.int32)
    cost = np.full((n_loc, K), np.inf)
    for k in range(n_loc):
        L = len(nbr_lists[k])
        nbr[k, :L] = nbr_lists[k]
        cost[k, :L] = cost_lists[k]
    valid_e = nbr >= 0
    np.seterr(invalid="ignore", divide="ignore")
    src = lid[path[0]]
    dst = lid[path[-1]]
    ve = valid_e.ravel()
    gcsr = csr_matrix((cost.ravel()[ve], (np.repeat(np.arange(n_loc), K)[ve], nbr.ravel()[ve])), shape=(n_loc, n_loc))
    exact_cost = float(dijkstra(gcsr, indices=src)[dst])   # reference: exact optimum inside the band
    if ap.guidance == "cost_to_go":
        pot = dijkstra(gcsr, indices=dst)                  # exact cost-to-go (edge costs are symmetric)
        pot = np.where(np.isfinite(pot), pot, 1e12)
    rc = np.where(valid_e, cost + pot[np.where(valid_e, nbr, 0)] - pot[:, None], np.inf)
    rc_norm = np.where(valid_e, np.maximum(rc, 0.0) / np.maximum(cost, 1e-9), np.inf)
    eta = np.where(valid_e, 1.0 / (rc_norm + 0.05), 0.0)
    eta_b = eta ** ap.beta

    seed_local = [lid[n] for n in path]

    def path_edges(loc_path):
        es = []
        for a, b in zip(loc_path[:-1], loc_path[1:]):
            slot = int(np.nonzero(nbr[a] == b)[0][0])
            es.append(a * K + slot)
        return np.array(es, dtype=np.int64)

    seed_edges = path_edges(seed_local)
    cost_flat = cost.ravel()
    seed_cost = float(cost_flat[seed_edges].sum())
    tau = np.ones((n_loc, K))
    tau_flat = tau.reshape(-1)  # view
    Q = seed_cost
    if ap.warm_start:
        tau_flat[seed_edges] += ap.seed_boost
    best_cost = seed_cost if ap.elitist else math.inf
    best_edges = seed_edges if ap.elitist else None
    best_loc = seed_local if ap.elitist else None
    A = ap.ants
    T = int(min(3000, 2.5 * len(path) + 100))
    ar = np.arange(A)
    hist = dict(iter_best=[], global_best=[], mean=[], success=[])
    for it in range(ap.iters):
        pos = np.full(A, src, np.int64)
        visited = np.zeros((A, n_loc), bool)
        visited[ar, src] = True
        steps_nodes = np.full((A, T + 1), -1, np.int32)
        steps_nodes[:, 0] = src
        steps_edges = np.full((A, T), -1, np.int64)
        alive = np.ones(A, bool)
        done = np.zeros(A, bool)
        for t in range(T):
            act = alive & ~done
            if not act.any():
                break
            cand = nbr[pos]
            ok = cand >= 0
            ok &= ~visited[ar[:, None], np.where(ok, cand, 0)]
            w = tau[pos] ** ap.alpha * eta_b[pos] if ap.alpha != 1.0 else tau[pos] * eta_b[pos]
            w = np.where(ok, w, 0.0)
            s = w.sum(axis=1)
            stuck = act & (s <= 0)
            alive[stuck] = False
            act = act & ~stuck
            if not act.any():
                break
            r = rng.random(A) * np.where(s > 0, s, 1.0)
            slot = np.minimum((np.cumsum(w, axis=1) < r[:, None]).sum(axis=1), K - 1)
            bad = ~ok[ar, slot]
            if bad.any():  # float round-off landed on a masked slot
                slot = np.where(bad, np.argmax(w, axis=1), slot)
            nxt = cand[ar, slot]
            steps_edges[act, t] = pos[act] * K + slot[act]
            pos = np.where(act, nxt, pos)
            visited[ar[act], pos[act]] = True
            steps_nodes[act, t + 1] = pos[act]
            done |= act & (pos == dst)
        # evaluate finished ants
        costs = np.full(A, np.inf)
        for a in np.nonzero(done)[0]:
            e = steps_edges[a][steps_edges[a] >= 0]
            costs[a] = cost_flat[e].sum()
        fin = np.isfinite(costs)
        tau *= (1.0 - ap.rho)
        np.maximum(tau, 1e-3, out=tau)
        depositors = [int(np.argmin(costs))] if (ap.best_only and fin.any()) else list(np.nonzero(fin)[0])
        for a in depositors:
            e = steps_edges[a][steps_edges[a] >= 0]
            np.add.at(tau_flat, e, Q / costs[a])
        if fin.any():
            ib = int(np.argmin(costs))
            if costs[ib] < best_cost:
                best_cost = float(costs[ib])
                best_edges = steps_edges[ib][steps_edges[ib] >= 0]
                best_loc = [int(x) for x in steps_nodes[ib][steps_nodes[ib] >= 0]]
            hist["iter_best"].append(float(costs[ib]))
            hist["mean"].append(float(costs[fin].mean()))
        else:
            hist["iter_best"].append(None)
            hist["mean"].append(None)
        if ap.elitist and best_edges is not None:
            np.add.at(tau_flat, best_edges, 1.0 * Q / best_cost)
        hist["global_best"].append(None if best_edges is None else float(best_cost))
        hist["success"].append(float(fin.mean()))
    if best_loc is None:
        return dict(path=None, cost=None, history=hist, seconds=time.perf_counter() - t0,
                    subgraph_nodes=n_loc, seed_cost=seed_cost, improved=False, exact_cost=exact_cost)
    gpath = [nodes[k] for k in best_loc]
    return dict(path=gpath, cost=best_cost, history=hist, seconds=time.perf_counter() - t0,
                subgraph_nodes=n_loc, seed_cost=seed_cost, improved=best_cost < seed_cost - 1e-6, exact_cost=exact_cost)


def _split_windows(grid, path, seg_km, Mall, hscale, tol=0.12):
    """Anchor nodes that cut the A* path into windows of at least seg_km.

    Anchors are only placed on CALM nodes (cost multiplier close to the minimum). An anchor
    inside a storm or risk zone would pin the route to that spot and stop the colony from
    detouring around the hazard, so a window simply stretches until the next calm node.
    """
    pts = [grid.node_latlon(n) for n in path]
    anchors = [0]
    acc = 0.0
    for k in range(1, len(path) - 3):
        acc += float(hav_km(pts[k - 1][0], pts[k - 1][1], pts[k][0], pts[k][1]))
        if acc >= seg_km and k - anchors[-1] >= 3 and Mall[path[k]] <= hscale + tol:
            anchors.append(k)
            acc = 0.0
    anchors.append(len(path) - 1)
    return anchors


def run_aco(grid: OceanGrid, Mall, astar_res, ap: AcoParams, hscale, seg_km=800.0, log=None):
    """Stage 2: refine the A* path window by window.

    One ant has to walk the whole route in a long thin band, which almost never succeeds, so
    the route is cut into windows between anchor nodes on the A* path and each window is
    optimised by its own colony. Windows are independent, which also makes them easy to
    re-run on their own (the 'residual-route replanning' idea in the paper).
    """
    t0 = time.perf_counter()
    path = astar_res["path"]
    anchors = _split_windows(grid, path, seg_km, Mall, hscale)
    rng = np.random.default_rng(ap.seed)
    out_path = [path[0]]
    wins = []
    total_seed = total_best = total_exact = 0.0
    sub_nodes = 0
    fallbacks = 0
    for w in range(len(anchors) - 1):
        seg = path[anchors[w]: anchors[w + 1] + 1]
        r = _aco_window(grid, Mall, seg, ap, hscale, rng)
        sub_nodes += r["subgraph_nodes"]
        total_seed += r["seed_cost"]
        total_exact += r["exact_cost"]
        if r["path"] is None:      # no ant finished and no incumbent: keep the A* piece
            fallbacks += 1
            r["path"], r["cost"] = seg, r["seed_cost"]
        total_best += r["cost"]
        out_path.extend(r["path"][1:])
        wins.append(r)
    iters = ap.iters
    hist = dict(iter_best=[], global_best=[], mean=[], success=[])
    for it in range(iters):
        gb, ib, mn, su = 0.0, 0.0, 0.0, []
        gb_ok = ib_ok = mn_ok = True
        for r in wins:
            h = r["history"]
            v = h["global_best"][it]
            gb += v if v is not None else r["seed_cost"]
            if h["iter_best"][it] is None:
                ib_ok = mn_ok = False
            else:
                ib += h["iter_best"][it]
                mn += h["mean"][it]
            su.append(h["success"][it])
        hist["global_best"].append(gb)
        hist["iter_best"].append(ib if ib_ok else None)
        hist["mean"].append(mn if mn_ok else None)
        hist["success"].append(float(np.mean(su)))
    return dict(path=out_path, cost=total_best, history=hist, seconds=time.perf_counter() - t0,
                subgraph_nodes=sub_nodes, seed_cost=total_seed, improved=total_best < total_seed - 1e-6,
                windows=len(wins), fallbacks=fallbacks, exact_cost=total_exact, anchors=[path[k] for k in anchors])


# ----------------------------------------------------------------------------- evaluation + smoothing
class Evaluator:
    def __init__(self, grid: OceanGrid, f: Fields, cp: CostParams):
        self.g, self.f, self.cp = grid, f, cp
        self.coast = np.clip(1.0 - grid.dist_land_km / cp.coast_buffer_km, 0.0, 1.0)
        self.base_mult = cp.w_dist + cp.w_fuel

    def _idx(self, la, lo):
        g = self.g
        i = np.clip(np.rint((la - LAT_MIN) / g.res).astype(np.int64), 0, g.H - 1)
        j = np.rint((lo + 180.0) / g.res).astype(np.int64) % g.W
        return g.near_i[i, j], g.near_j[i, j], i, j

    def seg(self, p, q, trusted=False):
        la, lo, d = lerp_segment(p, q)
        n = len(la)
        step = d / max(n - 1, 1)
        cp = self.cp
        if trusted:
            return dict(km=d, cost=d * self.base_mult, wx=0.0, risk_km=0.0, land=0, blocked=0, canal_km=d,
                        c_dist=cp.w_dist * d, c_wx=0.0, c_risk=0.0, c_fuel=cp.w_fuel * d, c_coast=0.0)
        land = int(globe.is_land(la, lo).sum())
        ni, nj, i, j = self._idx(la, lo)
        wx = self.f.wx[ni, nj].astype(np.float64)
        rk = self.f.rk[ni, nj].astype(np.float64)
        co = self.coast[ni, nj]
        blocked = int(self.f.blocked[i, j].sum())
        c_dist = cp.w_dist * d
        c_wx = cp.w_wx * float(wx.mean()) * d
        c_risk = cp.w_risk * cp.risk_pref * float(rk.mean()) * d
        c_fuel = cp.w_fuel * (1.0 + cp.sea_k * float(wx.mean())) * d
        c_coast = cp.w_coast * float(co.mean()) * d
        return dict(km=d, cost=c_dist + c_wx + c_risk + c_fuel + c_coast, wx=float(wx.mean()) * d,
                    risk_km=float((rk > 0).mean()) * d, land=land, blocked=blocked, canal_km=0.0,
                    c_dist=c_dist, c_wx=c_wx, c_risk=c_risk, c_fuel=c_fuel, c_coast=c_coast)

    def smooth(self, pts, trusted, window=40):
        n = len(pts)
        seg = [self.seg(pts[k], pts[k + 1], trusted[k]) for k in range(n - 1)]
        cum = np.concatenate([[0.0], np.cumsum([s["cost"] for s in seg])])
        keep = [0]
        i = 0
        while i < n - 1:
            best = i + 1
            if not trusted[i]:
                jmax = min(n - 1, i + window)
                for k in range(i, jmax):
                    if trusted[k]:
                        jmax = k
                        break
                for j in range(jmax, i + 1, -1):
                    st = self.seg(pts[i], pts[j])
                    if st["land"] == 0 and st["blocked"] == 0 and st["cost"] <= cum[j] - cum[i] + 1e-9:
                        best = j
                        break
            keep.append(best)
            i = best
        new_pts = [pts[k] for k in keep]
        new_tr = []
        for a, b in zip(keep[:-1], keep[1:]):
            new_tr.append(b == a + 1 and trusted[a])
        return new_pts, new_tr

    def summarize(self, pts, trusted):
        tot = dict(km=0.0, cost=0.0, wx=0.0, risk_km=0.0, land=0, blocked=0, canal_km=0.0,
                   c_dist=0.0, c_wx=0.0, c_risk=0.0, c_fuel=0.0, c_coast=0.0, samples=0)
        for k in range(len(pts) - 1):
            s = self.seg(pts[k], pts[k + 1], trusted[k])
            for key in tot:
                if key in s:
                    tot[key] += s[key]
            if not trusted[k]:
                tot["samples"] += len(lerp_segment(pts[k], pts[k + 1])[0])
        tot["mean_wx"] = tot["wx"] / max(tot["km"], 1e-9)
        return tot


def path_points(grid, path):
    pts = [grid.node_latlon(n) for n in path]
    trusted = [grid.edge_trusted(a, b) for a, b in zip(path[:-1], path[1:])]
    return pts, trusted


def unwrap_lons(pts):
    out, prev = [], None
    for la, lo in pts:
        if prev is not None:
            while lo - prev > 180:
                lo -= 360
            while prev - lo > 180:
                lo += 360
        out.append((la, lo))
        prev = lo
    return out


# ----------------------------------------------------------------------------- orchestration
def plan_route(grid, origin, dest, risk_zones, storms, restricted, cp: CostParams, ap: AcoParams,
               stage1_model="paper", smooth=True, run_aco_stage=True, log=None):
    log = log or (lambda s: None)
    out = dict(origin=origin, dest=dest)
    f = rasterize(grid, risk_zones, storms, restricted)
    out["fields"] = f
    Mc = composite_field(grid, f, cp)
    safer = cp.risk_pref > 1.0
    if stage1_model == "paper":
        Ma = astar_field(grid, f, safer)
        hs1 = 1.0
    else:
        Ma = Mc
        hs1 = cp.min_mult
    src_g, sla, slo, sd = grid.snap(*origin)
    dst_g, dla, dlo, dd = grid.snap(*dest)
    out["snap"] = dict(origin=(sla, slo, sd), dest=(dla, dlo, dd))
    if not np.isfinite(Ma.ravel()[src_g]) or not np.isfinite(Ma.ravel()[dst_g]):
        raise ValueError("Origin or destination sits inside a restricted zone")
    log("Stage 1: A* search...")
    Ma_all = flat_with_extras(grid, Ma, 1.0 if stage1_model == "paper" else cp.min_mult)
    a = astar(grid, Ma_all, src_g, dst_g, hs1)
    if a is None:
        raise ValueError("A* found no connected route between these ports on this graph")
    out["astar"] = a
    log(f"Stage 1 done: {len(a['path'])} nodes, {a['expanded']:,} expanded, {a['seconds']:.2f}s")
    if run_aco_stage:
        Mc_all = flat_with_extras(grid, Mc, cp.min_mult)
        log("Stage 2: ACO refinement window by window around the A* path...")
        aco = run_aco(grid, Mc_all, a, ap, cp.min_mult, seg_km=ap.seg_km)
        log(f"Stage 2 done: {aco['windows']} windows, {aco['subgraph_nodes']:,} band nodes, {aco['seconds']:.1f}s")
    else:
        aco = dict(path=None, cost=None, history=None, seconds=0.0, subgraph_nodes=0, seed_cost=None,
                   improved=False, windows=0, fallbacks=0, exact_cost=None, anchors=[])
    out["aco"] = aco
    ev = Evaluator(grid, f, cp)
    routes = {}
    keys = (("astar", a["path"]), ("aco", aco["path"] or a["path"])) if run_aco_stage else (("astar", a["path"]),)
    for key, pth in keys:
        pts, tr = path_points(grid, pth)
        raw = ev.summarize(pts, tr)
        if smooth:
            spts, str_ = ev.smooth(pts, tr)
        else:
            spts, str_ = pts, tr
        fin = ev.summarize(spts, str_)
        routes[key] = dict(pts=unwrap_lons(spts), trusted=str_, raw_pts=unwrap_lons(pts), stats=fin, raw_stats=raw)
    out["routes"] = routes
    out["cp"], out["ap"] = cp, ap
    return out
