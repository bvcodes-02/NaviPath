"""Static data for the NaviPath prototype.

Everything in the scenario presets is ILLUSTRATIVE DEMO DATA. It is not IMB, NAVAREA
or NOAA data. Replace it with real feeds when the project integrates them.
"""

# name -> (lat, lon). Ports are snapped to the nearest connected water node.
PORTS = {
    "Mumbai": (18.94, 72.84),
    "Chennai": (13.10, 80.30),
    "Kolkata (Haldia)": (22.03, 88.10),
    "Colombo": (6.95, 79.85),
    "Dubai (Jebel Ali)": (25.01, 55.06),
    "Jeddah": (21.48, 39.17),
    "Mombasa": (-4.06, 39.67),
    "Durban": (-29.87, 31.03),
    "Cape Town": (-33.91, 18.43),
    "Singapore": (1.27, 103.82),
    "Hong Kong": (22.30, 114.17),
    "Shanghai (Yangshan)": (30.63, 122.07),
    "Busan": (35.10, 129.04),
    "Tokyo": (35.62, 139.78),
    "Sydney": (-33.85, 151.21),
    "Los Angeles": (33.72, -118.27),
    "Balboa (Panama Pacific)": (8.95, -79.57),
    "Colon (Panama Atlantic)": (9.36, -79.90),
    "New York": (40.67, -74.04),
    "Santos": (-23.98, -46.30),
    "Rotterdam": (51.95, 4.14),
    "Valencia": (39.45, -0.32),
    "Genoa": (44.41, 8.92),
    "Piraeus": (37.94, 23.63),
}

# "Highway" corridors: hand-placed node chains through places a regular grid cannot
# resolve. They play the role of the TSS lane centerlines in the paper.
#   - canal=(a, b): segments a..b-1 are trusted (the 1 km land mask sees canals as land)
#   - every other segment is validated against the land mask when the graph is built
GATEWAYS = [
    dict(name="Suez Canal + Gulf of Suez",
         nodes=[(31.50, 32.36), (31.26, 32.31), (30.60, 32.34), (30.30, 32.38), (29.95, 32.55), (29.80, 32.56),
                (29.40, 32.68), (29.10, 32.87), (28.80, 32.98), (28.50, 33.14), (28.20, 33.42),
                (27.90, 33.75), (27.60, 34.00), (27.30, 34.40), (26.70, 35.00)],
         canal=(0, 5)),
    dict(name="Panama Canal (approximate)",
         nodes=[(9.50, -79.92), (9.36, -79.92), (9.28, -79.92), (9.20, -79.82), (9.08, -79.68),
                (8.99, -79.59), (8.93, -79.52), (8.80, -79.45), (8.60, -79.40)],
         canal=(0, 6)),
    dict(name="Strait of Gibraltar",
         nodes=[(36.15, -6.60), (36.00, -5.75), (35.97, -5.45), (36.10, -4.90), (36.40, -4.00)]),
    dict(name="Dover Strait",
         nodes=[(49.60, -3.00), (50.00, -1.00), (50.50, 0.50), (50.95, 1.40), (51.25, 2.00),
                (51.70, 2.60), (52.00, 3.40)]),
    dict(name="Bab-el-Mandeb",
         nodes=[(13.80, 42.50), (13.00, 43.00), (12.65, 43.35), (12.45, 43.60), (12.20, 44.20), (11.90, 45.00)]),
    dict(name="Strait of Hormuz",
         nodes=[(25.40, 57.50), (25.90, 56.90), (26.40, 56.60), (26.45, 55.80), (26.10, 55.00)]),
    dict(name="Malacca and Singapore Straits",
         nodes=[(6.00, 96.50), (5.60, 97.20), (5.20, 98.00), (4.80, 98.40), (4.20, 98.80), (3.00, 100.30),
                (2.30, 101.50), (1.70, 102.40), (1.35, 103.20), (1.14, 103.55), (1.14, 103.65),
                (1.16, 103.78), (1.18, 103.90), (1.20, 104.00), (1.20, 104.15), (1.20, 104.30),
                (1.20, 104.50)]),
]

# Illustrative risk zones (level 0.5 = medium, 1.0 = high). Boxes, not real polygons.
DEFAULT_RISK_ZONES = [
    dict(name="Gulf of Aden (illustrative high)", lat_min=11.0, lat_max=15.0, lon_min=43.0, lon_max=53.0, level=1.0),
    dict(name="Somali Basin (illustrative medium)", lat_min=-4.0, lat_max=11.0, lon_min=44.0, lon_max=60.0, level=0.5),
    dict(name="Gulf of Guinea (illustrative medium)", lat_min=0.0, lat_max=8.0, lon_min=-5.0, lon_max=10.0, level=0.5),
    dict(name="Malacca / Singapore Straits (illustrative medium)", lat_min=0.5, lat_max=6.0, lon_min=95.0, lon_max=105.0, level=0.5),
    dict(name="Sulu Sea (illustrative medium)", lat_min=4.0, lat_max=9.0, lon_min=118.0, lon_max=124.5, level=0.5),
]

# Synthetic weather cells (severity 0..1).
PRESETS = {
    "Calm seas (no storms)": dict(storms=[], restricted=[]),
    "Arabian Sea cyclone": dict(
        storms=[dict(name="Cyclone A", lat=16.5, lon=64.0, radius_km=650.0, severity=1.0)], restricted=[]),
    "North Atlantic gale": dict(
        storms=[dict(name="Gale N", lat=47.0, lon=-24.0, radius_km=650.0, severity=0.85)], restricted=[]),
    "Western Pacific typhoon": dict(
        storms=[dict(name="Typhoon P", lat=26.0, lon=150.0, radius_km=600.0, severity=0.9)], restricted=[]),
    "Cyclone + restricted exercise area": dict(
        storms=[dict(name="Cyclone A", lat=16.5, lon=64.0, radius_km=650.0, severity=1.0)],
        restricted=[dict(name="Naval exercise (NAVAREA-style)", lat_min=11.0, lat_max=14.0, lon_min=50.0, lon_max=55.0)]),
}
