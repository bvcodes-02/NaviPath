# NaviPath prototype (Python, Streamlit, Leaflet)

Two-stage maritime route planner: **A\*** on a land-masked ocean graph, then **ACO** refinement under the
composite cost C = w1*dist + w2*weather + w3*risk + w4*fuel (plus an optional coastal margin).

## Run

    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

The first run at each node spacing builds the land/water graph and caches it in `.cache/`
(0.25 deg takes about 10 s, 0.15 deg about 15 s and ~1.5 GB RAM). Later runs load in a second.
The browser needs internet for map tiles (Esri / OpenStreetMap).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, Leaflet map (folium), Altair charts |
| `engine.py` | Graph, land mask, A*, ACO, route evaluator, straightening |
| `data.py` | Ports, highway corridors, demo risk zones and storm presets |

## What fixes the land/water problem

1. **Real land mask.** Uses the 1 km GLOBE mask (`global-land-mask`), not a hand-drawn polygon.
2. **Edges are tested, not just nodes.** Each candidate edge is sampled about every 1 km; a single land sample rejects it.
3. **Density is selectable** (1.0, 0.5, 0.25, 0.15 deg). More nodes follow coasts and narrow passages better and put ports closer to the graph.
4. **Highway corridors** (Suez, Panama, Gibraltar, Dover, Bab-el-Mandeb, Hormuz, Malacca/Singapore) keep chokepoints open at any density.
   Suez and Panama are trusted canal segments because the mask sees canals as land. They are approximate.
5. **Verification.** Final routes are re-sampled at 1 km and the app reports land samples hit (expected 0).

## How the stages work

* **Stage 1, A\*:** by default the paper cost, distance x (1 + risk_weight). Optional: the full composite cost.
* **Stage 2, ACO:** runs window by window along the A\* path (anchors on calm nodes), inside a band of selectable width.
  Options: elitist seeding (A\* path as incumbent), pheromone warm-start, iteration-best deposit, and the ants' heuristic
  information (straight-line bound or exact in-band cost-to-go). The convergence chart shows the A\* incumbent and the exact in-band optimum.

## Limits to keep in mind

* Weather, storms, piracy and restricted areas are **synthetic demo data**. `rasterize()` in `engine.py` is the place to plug in GFS, IMB and NAVAREA feeds.
* A\* is optimal for its own cost, so ACO only helps when the stage costs differ and a hazard sits near the route.
* Latitude range is -70 to 75 deg. Ports snap to the nearest connected water node.
* Prototype numbers are for exploration. Do not quote them as paper results without a controlled evaluation.

## Screenshot for the paper (Fig. 5)

Collapse the sidebar, pick the light-gray base map, compute a route between two ports, and capture the map area.
