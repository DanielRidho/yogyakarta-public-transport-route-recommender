import math
from typing import List, Dict

# ====================== PARAMETER DASAR ======================
WALK_SPEED_M_PER_MIN = 80.0        # kecepatan jalan kaki ~4.8 km/jam
MAX_ACCESS_WALK_M    = 600.0       # radius maksimal akses ke halte terdekat
MAX_DIRECT_WALK_M    = 800.0       # radius maksimal jalan langsung origin→destination
MIN_WALK_TIME_MIN    = 0.2         # waktu minimum supaya tidak 0
DEFAULT_TOP_N        = 4           # jumlah halte akses default

# ====================== UTILITAS GEOSPASIAL ======================
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _walk_time_min(distance_m: float) -> float:
    return max(MIN_WALK_TIME_MIN, round(distance_m / WALK_SPEED_M_PER_MIN, 2))

# ====================== AKSES USER KE HALTE ======================
def generate_access_edges(
    user_node_id: str,
    lat: float,
    lon: float,
    stops_df,
    top_n: int = DEFAULT_TOP_N,
    max_access_walk_m: float = MAX_ACCESS_WALK_M
) -> List[Dict]:
    candidates = []
    for _, stop in stops_df.iterrows():
        dist = _haversine(lat, lon, float(stop["lat"]), float(stop["lon"]))
        if dist <= max_access_walk_m:
            candidates.append((stop["stop_id"], dist))

    # Urut halte berdasarkan jarak dan ambil N terdekat
    candidates.sort(key=lambda x: x[1])
    selected = candidates[:max(1, int(top_n))]

    edges: List[Dict] = []
    for stop_id, dist in selected:
        edges.append({
            "u_stop_id": user_node_id,
            "v_stop_id": stop_id,
            "mode": "walk",
            "distance_m": float(dist),
            "travel_time_min_base": _walk_time_min(dist),
            "fare_id": "F_WALK",
            "transfer_penalty_min": 0.0,
            "route_id": None
        })

    return edges

# ====================== DIRECT WALK ORIGIN → DEST ======================
def generate_direct_walk_edge(
    origin_node_id: str,
    origin_lat: float,
    origin_lon: float,
    dest_node_id: str,
    dest_lat: float,
    dest_lon: float,
    max_direct_walk_m: float = MAX_DIRECT_WALK_M
) -> List[Dict]:
    distance = _haversine(origin_lat, origin_lon, dest_lat, dest_lon)
    if distance <= max_direct_walk_m:
        return [{
            "u_stop_id": origin_node_id,
            "v_stop_id": dest_node_id,
            "mode": "walk",
            "distance_m": float(distance),
            "travel_time_min_base": _walk_time_min(distance),
            "fare_id": "F_WALK",
            "transfer_penalty_min": 0.0,
            "route_id": None
        }]
    return []
