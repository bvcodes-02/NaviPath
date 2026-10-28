"""NaviPath prototype UI (Streamlit + Leaflet via folium + Altair charts).

Run:  streamlit run app.py
"""
import math
import os
import sys

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import engine as E  # noqa: E402

st.set_page_config(page_title="NaviPath prototype", page_icon="🧭", layout="wide")

RES_OPTIONS = {
    "1.0°  coarse (instant)": 1.0,
    "0.5°  balanced": 0.5,
    "0.25°  dense (recommended)": 0.25,
    "0.15°  very dense (slow first build, ~1.5 GB RAM)": 0.15,
}
TILES = {
    "Light gray (Esri)": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                          "Esri, HERE, Garmin", 14),
    "Ocean (Esri)": ("https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
                     "Esri, GEBCO, NOAA, National Geographic", 10),
    "Satellite (Esri)": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         "Esri, Maxar, Earthstar Geographics", 14),
    "Street (OpenStreetMap)": ("OpenStreetMap", None, None),
}
C_ASTAR, C_ACO = "#1f4e9c", "#d62728"


@st.cache_resource(show_spinner=False)
def load_grid(res: float):
    return E.get_grid(res, D.GATEWAYS)


def records(df: pd.DataFrame, numeric):
    out = []
    for r in df.to_dict("records"):
        try:
            rec = {k: (float(r[k]) if k in numeric else str(r[k])) for k in r}
        except (TypeError, ValueError):
            continue
        if any(isinstance(v, float) and math.isnan(v) for v in rec.values()):
            continue
        out.append(rec)
    return out


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧭 NaviPath")
    st.caption("Prototype: A* + ACO route planning on a land-masked ocean graph.")
    ports = list(D.PORTS)
    st.subheader("Route")
    origin = st.selectbox("Origin port", ports, index=ports.index("Mumbai"))
    dest = st.selectbox("Destination port", ports, index=ports.index("Rotterdam"))

    st.subheader("Graph density")
    res_label = st.selectbox("Node spacing (lattice)", list(RES_OPTIONS), index=2,
                             help="Smaller spacing means more nodes. Land is always tested on every edge at ~1 km, "
                                  "so density changes how closely the graph follows coasts and narrow passages.")
    res = RES_OPTIONS[res_label]

    st.subheader("Scenario (synthetic demo data)")
    preset = st.selectbox("Weather / restriction preset", list(D.PRESETS), index=1)
    use_risk = st.checkbox("Include illustrative piracy / risk zones", True)
    with st.expander("Edit storms, risk zones, restricted areas"):
        st.caption("Storms: severity 0 to 1. Risk zones: level 0.5 medium, 1.0 high. Boxes are illustrative, not IMB data.")
        storm_df = st.data_editor(
            pd.DataFrame(D.PRESETS[preset]["storms"], columns=["name", "lat", "lon", "radius_km", "severity"]),
            num_rows="dynamic", key=f"storms_{preset}", width="stretch",
            column_config={c: st.column_config.NumberColumn(c) for c in ("lat", "lon", "radius_km", "severity")})
        risk_df = st.data_editor(
            pd.DataFrame(D.DEFAULT_RISK_ZONES, columns=["name", "lat_min", "lat_max", "lon_min", "lon_max", "level"]),
            num_rows="dynamic", key="risk_zones", width="stretch",
            column_config={c: st.column_config.NumberColumn(c) for c in ("lat_min", "lat_max", "lon_min", "lon_max", "level")})
        restr_df = st.data_editor(
            pd.DataFrame(D.PRESETS[preset]["restricted"], columns=["name", "lat_min", "lat_max", "lon_min", "lon_max"]),
            num_rows="dynamic", key=f"restr_{preset}", width="stretch",
            column_config={c: st.column_config.NumberColumn(c) for c in ("lat_min", "lat_max", "lon_min", "lon_max")})

    st.subheader("Cost model")
    pref = st.radio("Risk preference", ["OPTIMAL", "SAFER_OPTIMAL"], horizontal=True,
                    help="SAFER_OPTIMAL makes risk zones 3x more expensive (and adds risk to the stage-1 A* cost).")
    stage1 = st.radio("Stage-1 A* cost", ["Paper: distance x (1 + risk)", "Full composite (same as ACO)"],
                      help="With the full composite cost A* is already optimal on the graph, so ACO has little to add. "
                           "ACO refines the route mainly when stage 1 uses the simpler cost.")
    with st.expander("Weights  C = w1*dist + w2*weather + w3*risk + w4*fuel"):
        w1 = st.slider("w1 distance", 0.0, 1.0, 0.4, 0.05)
        w2 = st.slider("w2 weather", 0.0, 1.0, 0.3, 0.05)
        w3 = st.slider("w3 risk", 0.0, 1.0, 0.2, 0.05)
        w4 = st.slider("w4 fuel proxy", 0.0, 1.0, 0.1, 0.05)
        w5 = st.slider("coastal margin (prototype extra)", 0.0, 0.5, 0.1, 0.05)
        sea_k = st.slider("rough-sea fuel penalty", 0.0, 3.0, 1.0, 0.1)

    with st.expander("ACO settings"):
        ants = st.slider("Ants per window", 5, 100, 30)
        iters = st.slider("Iterations", 10, 200, 60)
        alpha = st.slider("alpha (pheromone weight)", 0.5, 3.0, 1.0, 0.1)
        beta = st.slider("beta (heuristic weight)", 0.0, 5.0, 2.0, 0.1)
        rho = st.slider("rho (evaporation)", 0.01, 0.6, 0.10, 0.01)
        band_km = st.slider("Search band half-width (km)", 100, 1500, 600, 50)
        seg_km = st.slider("ACO window length (km)", 300, 3000, 800, 100)
        guide_label = st.selectbox(
            "Ant heuristic information", ["Straight-line lower bound (A* style)", "Exact cost-to-go in band (strong guidance)"],
            help="The lower bound is weak: ants find small gains. Exact cost-to-go lets the colony reach the best route "
                 "inside the band, but it also shows that a plain search on the composite cost would do the same job.")
        elitist = st.checkbox("Elitist seeding (A* path kept as incumbent)", True)
        warm = st.checkbox("Pheromone warm-start from A* path", True)
        best_only = st.checkbox("Only iteration-best ant deposits", True)
        seed = st.number_input("Random seed", 0, 10_000, 7)

    st.subheader("Display")
    tile_label = st.selectbox("Base map", list(TILES), index=0)
    smooth = st.checkbox("Straighten routes (water-checked shortcuts)", True)
    show_nodes = st.checkbox("Show graph nodes (thinned)", False)
    show_corr = st.checkbox("Show highway corridors", True)
    run = st.button("Compute route", type="primary", width="stretch")


def make_params():
    cp = E.CostParams(w_dist=w1, w_wx=w2, w_risk=w3, w_fuel=w4, w_coast=w5,
                      risk_pref=3.0 if pref == "SAFER_OPTIMAL" else 1.0, sea_k=sea_k)
    ap = E.AcoParams(ants=ants, iters=iters, alpha=alpha, beta=beta, rho=rho, band_km=float(band_km),
                     seg_km=float(seg_km), elitist=elitist, warm_start=warm, best_only=best_only, seed=int(seed),
                     guidance="cost_to_go" if guide_label.startswith("Exact") else "bound")
    return cp, ap


def scenario():
    storms = records(storm_df, {"lat", "lon", "radius_km", "severity"})
    zones = records(risk_df, {"lat_min", "lat_max", "lon_min", "lon_max", "level"}) if use_risk else []
    restr = records(restr_df, {"lat_min", "lat_max", "lon_min", "lon_max"})
    return storms, zones, restr


# --------------------------------------------------------------------------- map
def near(lon, ref):
    return lon + 360.0 * round((ref - lon) / 360.0)


def build_map(res_, grid, tile_name, show_nodes_, show_corr_):
    ra, rc = res_["routes"]["astar"]["pts"], res_["routes"]["aco"]["pts"]
    allp = ra + rc
    lats = [p[0] for p in allp]
    lons = [p[1] for p in allp]
    clon = (min(lons) + max(lons)) / 2
    m = folium.Map(location=[(min(lats) + max(lats)) / 2, clon], zoom_start=3, tiles=None,
                   control_scale=True, world_copy_jump=True)
    url, attr, zmax = TILES[tile_name]
    if attr is None:
        folium.TileLayer(url, name="Base map").add_to(m)
    else:
        folium.TileLayer(url, attr=attr, name="Base map", max_zoom=zmax).add_to(m)

    g_zone = folium.FeatureGroup("Risk zones (illustrative)").add_to(m)
    for z in res_["scenario"]["zones"]:
        off = near(z["lon_min"], clon) - z["lon_min"]
        col = "#d62728" if z["level"] >= 1.0 else "#ff9800"
        folium.Rectangle([[z["lat_min"], z["lon_min"] + off], [z["lat_max"], z["lon_max"] + off]], color=col,
                         weight=1, fill=True, fill_opacity=0.16, tooltip=f"{z['name']} (level {z['level']})").add_to(g_zone)
    g_storm = folium.FeatureGroup("Storm cells (synthetic)").add_to(m)
    for s in res_["scenario"]["storms"]:
        folium.Circle([s["lat"], near(s["lon"], clon)], radius=s["radius_km"] * 1000, color="#6a1b9a", weight=1,
                      fill=True, fill_opacity=0.18, tooltip=f"{s['name']} severity {s['severity']}").add_to(g_storm)
    g_res = folium.FeatureGroup("Restricted areas").add_to(m)
    for z in res_["scenario"]["restricted"]:
        off = near(z["lon_min"], clon) - z["lon_min"]
        folium.Rectangle([[z["lat_min"], z["lon_min"] + off], [z["lat_max"], z["lon_max"] + off]], color="#444",
                         weight=2, dash_array="4 4", fill=True, fill_opacity=0.25, tooltip=z["name"]).add_to(g_res)

    if show_corr_:
        g_c = folium.FeatureGroup("Highway corridors").add_to(m)
        for gw in grid.gateway_lines:
            pts = [(la, near(lo, clon)) for la, lo in gw["nodes"]]
            folium.PolyLine(pts, color="#777", weight=2, opacity=0.9, dash_array="2 5", tooltip=gw["name"]).add_to(g_c)
    if show_nodes_:
        g_n = folium.FeatureGroup("Graph nodes (thinned)").add_to(m)
        r = grid.res
        i0 = max(0, int((min(lats) - 3 - E.LAT_MIN) / r))
        i1 = min(grid.H - 1, int((max(lats) + 3 - E.LAT_MIN) / r))
        j0 = int((min(lons) - 3 + 180) / r)
        j1 = int((max(lons) + 3 + 180) / r)
        total = max(1, (i1 - i0 + 1) * (j1 - j0 + 1))
        stride = max(1, int(math.ceil(math.sqrt(total / 2500.0))))
        for i in range(i0, i1 + 1, stride):
            for j in range(j0, j1 + 1, stride):
                if grid.main_mask[i * grid.W + (j % grid.W)]:
                    folium.CircleMarker([float(grid.lat[i]), -180.0 + j * r], radius=1.3, weight=0, fill=True,
                                        fill_color="#2b6cb0", fill_opacity=0.75).add_to(g_n)

    folium.PolyLine(ra, color=C_ASTAR, weight=3, opacity=0.95, dash_array="9 7",
                    tooltip="Stage 1: A* route").add_to(folium.FeatureGroup("A* route (stage 1)").add_to(m))
    folium.PolyLine(rc, color=C_ACO, weight=4, opacity=0.95,
                    tooltip="Stage 2: A* + ACO route").add_to(folium.FeatureGroup("A* + ACO route (stage 2)").add_to(m))

    g_p = folium.FeatureGroup("Ports").add_to(m)
    o_lat, o_lon = rc[0]
    d_lat, d_lon = rc[-1]
    for (la, lo), name, col in ((rc[0], res_["origin_name"], "#2e7d32"), (rc[-1], res_["dest_name"], "#b71c1c")):
        folium.CircleMarker([la, lo], radius=7, color="#000", weight=1.5, fill=True, fill_color=col,
                            fill_opacity=1, tooltip=folium.Tooltip(name, permanent=True, direction="top")).add_to(g_p)
    m.fit_bounds([[min(lats) - 2, min(lons) - 2], [max(lats) + 2, max(lons) + 2]])
    folium.LayerControl(collapsed=True).add_to(m)
    legend = f"""
    <div style="position:absolute;bottom:26px;left:12px;z-index:9999;background:rgba(255,255,255,.93);
         padding:7px 10px;border:1px solid #999;border-radius:4px;font:12px/1.55 'Times New Roman',serif;color:#111">
      <b>NaviPath route</b><br>
      <span style="display:inline-block;width:28px;border-top:3px dashed {C_ASTAR};vertical-align:middle"></span> Stage 1: A*<br>
      <span style="display:inline-block;width:28px;border-top:4px solid {C_ACO};vertical-align:middle"></span> Stage 2: A* + ACO<br>
      <span style="display:inline-block;width:14px;height:10px;background:#ff9800;opacity:.5;vertical-align:middle"></span> Risk zone
      <span style="display:inline-block;width:14px;height:10px;background:#6a1b9a;opacity:.4;vertical-align:middle;margin-left:6px"></span> Storm<br>
      <span style="display:inline-block;width:28px;border-top:2px dotted #777;vertical-align:middle"></span> Highway corridor
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m


# --------------------------------------------------------------------------- charts
def breakdown_chart(res_):
    rows = []
    for key, label in (("astar", "A*"), ("aco", "A* + ACO")):
        s = res_["routes"][key]["stats"]
        for comp, name in (("c_dist", "Distance"), ("c_wx", "Weather"), ("c_risk", "Risk"), ("c_fuel", "Fuel proxy"),
                           ("c_coast", "Coastal margin")):
            rows.append(dict(route=label, component=name, value=s[comp]))
    df = pd.DataFrame(rows)
    return (alt.Chart(df).mark_bar().encode(
        y=alt.Y("route:N", title=None, sort=["A*", "A* + ACO"]),
        x=alt.X("value:Q", title="Composite cost (stacked terms of C)"),
        color=alt.Color("component:N", title="Term", scale=alt.Scale(
            domain=["Distance", "Weather", "Risk", "Fuel proxy", "Coastal margin"],
            range=["#4c78a8", "#9467bd", "#e45756", "#f2a33a", "#8c8c8c"])),
        tooltip=["route", "component", alt.Tooltip("value:Q", format=".0f")]).properties(height=150))


def convergence_chart(res_):
    h = res_["aco"]["history"]
    n = len(h["global_best"])
    df = pd.DataFrame({"iteration": range(1, n + 1), "Best so far": h["global_best"],
                       "Iteration best": h["iter_best"], "Colony mean": h["mean"]})
    long = df.melt("iteration", var_name="series", value_name="cost")
    base = alt.Chart(long).mark_line(point=False).encode(
        x=alt.X("iteration:Q", title="Iteration"), y=alt.Y("cost:Q", title="Sum of window costs", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, scale=alt.Scale(
            domain=["Best so far", "Iteration best", "Colony mean"], range=[C_ACO, "#f2a33a", "#8c8c8c"])),
        strokeDash=alt.StrokeDash("series:N", legend=None))
    rules = pd.DataFrame({"y": [res_["aco"]["seed_cost"], res_["aco"]["exact_cost"]],
                          "label": ["A* path (incumbent)", "Exact optimum inside band (Dijkstra)"]})
    r = alt.Chart(rules).mark_rule(strokeDash=[5, 4]).encode(y="y:Q", color=alt.Color("label:N", title=None, scale=alt.Scale(
        range=[C_ASTAR, "#2e7d32"])))
    return (base + r).properties(height=320)


# --------------------------------------------------------------------------- compute
if run:
    if origin == dest:
        st.error("Pick two different ports.")
    else:
        cp, ap = make_params()
        storms, zones, restr = scenario()
        try:
            with st.status("Planning route...", expanded=True) as status:
                st.write(f"Loading the {res}° ocean graph (first build is cached to disk)...")
                grid = load_grid(res)
                result = E.plan_route(grid, D.PORTS[origin], D.PORTS[dest], zones, storms, restr, cp, ap,
                                      stage1_model="paper" if stage1.startswith("Paper") else "full",
                                      smooth=smooth, log=st.write)
                result["scenario"] = dict(storms=storms, zones=zones, restricted=restr)
                result["origin_name"], result["dest_name"] = origin, dest
                result["res"] = res
                st.session_state["result"] = result
                st.session_state.pop("lab", None)
                status.update(label="Route ready", state="complete", expanded=False)
        except ValueError as e:
            st.error(str(e))

result = st.session_state.get("result")
st.title("NaviPath route planner")
st.caption("Stage 1: A* on a land-masked ocean graph.  Stage 2: ACO refines the route under the composite cost.  "
           "Weather, storms and risk zones here are synthetic demo data.")

if result is None:
    st.info("Pick two ports in the sidebar and press **Compute route**.")
    st.stop()

grid = load_grid(result["res"])
A, C = result["routes"]["astar"]["stats"], result["routes"]["aco"]["stats"]
aco = result["aco"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Route length (A* + ACO)", f"{C['km']:,.0f} km", f"{C['km'] - A['km']:+,.0f} km vs A*", delta_color="off")
m2.metric("Composite cost", f"{C['cost']:,.0f}", f"{(C['cost'] / A['cost'] - 1) * 100:+.2f}% vs A*", delta_color="inverse")
m3.metric("Land crossings (1 km samples)", f"{C['land']} of {C['samples']:,}")
m4.metric("Graph", f"{grid.stats()['connected_nodes']:,} nodes", f"{result['res']}° spacing", delta_color="off")

tile_name = tile_label
fmap = build_map(result, grid, tile_name, show_nodes, show_corr)
st_folium(fmap, height=640, use_container_width=True, returned_objects=[], key="map")

t1, t2, t3, t4 = st.tabs(["Route comparison", "ACO convergence", "Graph and land/water check", "Notes"])

with t1:
    def fmt(v, d=0):
        return f"{v:,.{d}f}"
    rows = [
        ("Route length (km)", fmt(A["km"]), fmt(C["km"])),
        ("Composite cost C (lower is better)", fmt(A["cost"]), fmt(C["cost"])),
        ("Weather exposure (severity x km)", fmt(A["wx"]), fmt(C["wx"])),
        ("Distance inside risk zones (km)", fmt(A["risk_km"]), fmt(C["risk_km"])),
        ("Canal / corridor distance (km)", fmt(A["canal_km"]), fmt(C["canal_km"])),
        ("Land samples hit / checked", f"{A['land']} / {A['samples']:,}", f"{C['land']} / {C['samples']:,}"),
        ("Restricted-area samples hit", str(A["blocked"]), str(C["blocked"])),
        ("Search effort", f"{result['astar']['expanded']:,} nodes expanded, {result['astar']['seconds']:.2f} s",
         f"{aco['windows']} windows, {aco['subgraph_nodes']:,} band nodes, {aco['seconds']:.1f} s"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Stage 1: A*", "Stage 2: A* + ACO"]), hide_index=True, width="stretch")
    st.altair_chart(breakdown_chart(result), width="stretch")
    gap = (aco["cost"] / aco["exact_cost"] - 1) * 100 if aco["exact_cost"] else 0
    gain = (1 - aco["cost"] / aco["seed_cost"]) * 100
    st.caption(f"On the graph itself ACO changed the summed window cost by {-gain:+.2f}% relative to the A* path. "
               f"The exact optimum inside the same search band (Dijkstra) is {gap:.2f}% below what ACO reached, "
               "so ACO is shown against a true reference here. Displayed route lengths and costs are measured on the "
               "straightened polyline, not the raw graph path.")

with t2:
    st.altair_chart(convergence_chart(result), width="stretch")
    h = aco["history"]
    st.caption(f"Ants finishing a window: {np.mean(h['success']) * 100:.0f}% on average. "
               f"Windows without any finishing ant fall back to the A* piece: {aco['fallbacks']}. "
               "Turn off elitist seeding and warm-start to see the colony working on its own.")

with t3:
    s = grid.stats()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Graph**")
        st.dataframe(pd.DataFrame({"Item": ["Lattice spacing", "Grid", "Water nodes", "Connected ocean nodes",
                                            "Edges (land-checked)", "Highway corridors", "Corridor nodes"],
                                   "Value": [f"{s['res']}°", s["grid"], f"{s['water_nodes']:,}", f"{s['connected_nodes']:,}",
                                             f"{s['edges']:,}", str(s["corridors"]), str(s["corridor_nodes"])]}),
                     hide_index=True, width="stretch")
        so, sd = result["snap"]["origin"], result["snap"]["dest"]
        st.caption(f"Ports are snapped to the nearest connected water node: origin {so[2]:.0f} km, destination {sd[2]:.0f} km. "
                   "Snap distances shrink as density rises.")
        for w in s["warnings"]:
            st.warning(w)
    with c2:
        st.markdown("**Highway corridors**")
        st.dataframe(pd.DataFrame([dict(Corridor=g["name"], Nodes=len(g["nodes"]),
                                        Type="canal (trusted)" if g.get("canal") else "strait (land-checked)")
                                   for g in grid.gateway_lines]), hide_index=True, width="stretch")

    st.markdown("**Does a denser graph help? Resolution comparison (A* only, same scenario)**")
    labs = st.multiselect("Spacings to compare", list(RES_OPTIONS), default=list(RES_OPTIONS)[:3])
    if st.button("Run comparison"):
        cp, ap = make_params()
        storms_, zones_, restr_ = result["scenario"]["storms"], result["scenario"]["zones"], result["scenario"]["restricted"]
        out = []
        with st.status("Comparing graph densities...", expanded=True) as stt:
            for lab in labs:
                r_ = RES_OPTIONS[lab]
                st.write(f"{r_}° ...")
                g_ = load_grid(r_)
                try:
                    rr = E.plan_route(g_, D.PORTS[result["origin_name"]], D.PORTS[result["dest_name"]], zones_, storms_,
                                      restr_, cp, ap, stage1_model="paper" if stage1.startswith("Paper") else "full",
                                      smooth=smooth, run_aco_stage=False)
                    a_ = rr["routes"]["astar"]["stats"]
                    out.append({"Spacing": f"{r_}°", "Connected nodes": g_.stats()["connected_nodes"],
                                "Edges": g_.stats()["edges"], "Port snap (km)": f"{rr['snap']['origin'][2]:.0f} / {rr['snap']['dest'][2]:.0f}",
                                "Route (km)": round(a_["km"]), "Land samples hit": a_["land"],
                                "Nodes expanded": rr["astar"]["expanded"], "A* time (s)": round(rr["astar"]["seconds"], 2)})
                except ValueError as e:
                    out.append({"Spacing": f"{r_}°", "Connected nodes": g_.stats()["connected_nodes"], "Edges": g_.stats()["edges"],
                                "Port snap (km)": "-", "Route (km)": None, "Land samples hit": None,
                                "Nodes expanded": None, "A* time (s)": None})
            stt.update(label="Comparison ready", state="complete", expanded=False)
        st.session_state["lab"] = pd.DataFrame(out)
    if "lab" in st.session_state:
        st.dataframe(st.session_state["lab"], hide_index=True, width="stretch")
        st.caption("Land is checked on every edge at every density, so crossings stay at zero. What density changes is "
                   "how close ports sit to the graph, how well narrow passages are represented and how smooth the path is. "
                   "Very coarse graphs can lose a passage altogether and send the route the long way round.")

with t4:
    st.markdown(
        """
**How land and water are separated.** Every candidate edge is sampled about every 1 km against the GLOBE 1 km land
mask. One land sample rejects the edge, so a route cannot cut a headland even when both end nodes are in the water.
Straightened routes are re-checked the same way, and the land-crossing counter above is computed on the final polyline.

**Why the graph is denser.** Node spacing is selectable down to 0.15° (about 17 km). Hand-placed highway corridors
(Suez, Panama, Gibraltar, Dover, Bab-el-Mandeb, Hormuz, Malacca/Singapore) stand in for TSS lane centerlines and keep
narrow passages open at any density. Suez and Panama are trusted canal segments because the 1 km mask treats them as land.

**Stage 2.** ACO works window by window along the A* path. Anchors sit on calm nodes so a storm cannot pin the route.
Pheromone can be warm-started from the A* path, and the A* path can be kept as the incumbent, which guarantees the
result is never costlier than A* under the composite cost.

**Why gains vary.** ACO can only beat A* when the composite cost differs from the stage-1 cost and a storm or risk zone
sits near the route. With the paper weights the gain is small. Raise the weather weight and the rough-sea penalty, or switch
the ants to exact cost-to-go guidance, to see larger differences.

**Read results with care.** Weather, storms and risk boxes are synthetic. A* is optimal for its own cost, so ACO only
helps when stage 1 uses a simpler cost than stage 2, and the gain is often small. Do not quote prototype numbers as paper results.
"""
    )
