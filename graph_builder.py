import math
from typing import Optional, Iterable, Dict, Tuple
import networkx as nx
import pandas as pd

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _auto_crosswalks(stops_df: pd.DataFrame, threshold_m: float = 120.0) -> pd.DataFrame:
    out = []
    pts = stops_df[["stop_id", "lat", "lon"]].dropna().to_dict("records")
    for i in range(len(pts)):
        a = pts[i]
        for j in range(i + 1, len(pts)):
            b = pts[j]
            d = _haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            if d <= threshold_m:
                tmin = max(0.2, round(d / 80.0, 2))
                for u, v in [(a, b), (b, a)]:
                    out.append(
                        dict(
                            u_stop_id=u["stop_id"],
                            v_stop_id=v["stop_id"],
                            mode="walk",
                            distance_m=d,
                            travel_time_min_base=tmin,
                            fare_id="F_WALK",
                            walking=True,
                            route_id=None,
                            transfer_penalty_min=0.0,
                        )
                    )
    return pd.DataFrame(out)


# ============================ POSISI NODE ============================

def _attach_node_pos(G: nx.DiGraph, stops_df: pd.DataFrame) -> None:
    for _, row in stops_df.dropna(subset=["stop_id", "lat", "lon"]).iterrows():
        sid = row["stop_id"]
        if not G.has_node(sid):
            G.add_node(sid)
        lat = float(row["lat"])
        lon = float(row["lon"])
        G.nodes[sid]["lat"] = lat
        G.nodes[sid]["lon"] = lon
        G.nodes[sid]["pos"] = (lat, lon)


def _set_node_pos_if_known(G: nx.DiGraph, stop_id, stops_df: pd.DataFrame) -> None:
    if stop_id is None:
        return
    if not G.has_node(stop_id):
        G.add_node(stop_id)
    n = G.nodes[stop_id]
    if "pos" not in n:
        m = stops_df.loc[stops_df["stop_id"] == stop_id]
        if not m.empty:
            lat = float(m.iloc[0].get("lat"))
            lon = float(m.iloc[0].get("lon"))
            n["lat"], n["lon"], n["pos"] = lat, lon, (lat, lon)


# ============================ STOP SEQUENCE ============================

def _build_stop_sequence_lookup(timetables_df: Optional[pd.DataFrame]):
    if timetables_df is None or timetables_df.empty:
        return {}
    if not {"route_id", "stop_id", "stop_sequence"}.issubset(timetables_df.columns):
        return {}
    grp = timetables_df.dropna(subset=["route_id", "stop_id", "stop_sequence"])
    return grp.groupby(["route_id", "stop_id"])["stop_sequence"].min().to_dict()


# ============================ GRAPH BUILDER ============================

def build_graph(
    edges_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    timetables_df: Optional[pd.DataFrame] = None,
    access_edges: Optional[Iterable[dict]] = None,
    crosswalk_threshold_m: float = 120.0,
    traffic_rules_df: Optional[pd.DataFrame] = None,
):

    # ================= 1) PREPROCESS TRAFFIC RULES ===================
    traffic_map = {}  # route_id → list faktor
    if traffic_rules_df is not None and not traffic_rules_df.empty:
        for _, r in traffic_rules_df.iterrows():
            rid = r["route_id"]
            if rid not in traffic_map:
                traffic_map[rid] = []
            traffic_map[rid].append(float(r["delay_factor"]))

    def apply_traffic(rid, base_time):
        if rid not in traffic_map:
            return base_time
        avg = sum(traffic_map[rid]) / len(traffic_map[rid])
        return base_time * avg

    # ================= 2) CROSSWALK OTOMATIS ==========================
    cross = _auto_crosswalks(stops_df, crosswalk_threshold_m)
    if not cross.empty:
        edges_df = pd.concat([edges_df, cross], ignore_index=True)

    # ================= 3) STOP SEQUENCE UNTUK KERETA ==================
    stop_seq_lookup = _build_stop_sequence_lookup(timetables_df)
    rail_modes = {"krl", "prameks", "railink"}

    # ================= 4) BUILD GRAPH ============================
    G = nx.DiGraph()
    _attach_node_pos(G, stops_df)

    for _, e in edges_df.iterrows():
        u = e["u_stop_id"]
        v = e["v_stop_id"]
        mode = e.get("mode", "bus")
        rid = e.get("route_id", None)
        edir_val = e.get("direction", None)

        # Filter direction
        if isinstance(edir_val, str):
            edir = edir_val.strip()
            if edir and edir not in ("A", "B"):
                continue

        # Batasi arah kereta follow sequence
        if (
            mode in rail_modes
            and rid is not None
            and stop_seq_lookup
        ):
            seq_u = stop_seq_lookup.get((rid, u))
            seq_v = stop_seq_lookup.get((rid, v))
            if seq_u is not None and seq_v is not None:
                if seq_v <= seq_u:
                    continue

        base_tt = float(e.get("travel_time_min_base", 0.0))
        base_tt = apply_traffic(rid, base_tt)

        travel = base_tt + float(e.get("transfer_penalty_min", 0.0) or 0.0)

        _set_node_pos_if_known(G, u, stops_df)
        _set_node_pos_if_known(G, v, stops_df)

        G.add_edge(
            u, v,
            travel_time=travel,
            travel_time_min_base=base_tt,
            transfer_penalty_min=float(e.get("transfer_penalty_min", 0.0) or 0.0),
            headway_min=float(e.get("headway_min", 0.0) or 0.0),
            distance_m=float(e.get("distance_m", 0.0) or 0.0),
            mode=mode,
            route_id=rid if pd.notna(rid) else None,
            fare_id=e.get("fare_id", None),
            direction=edir_val if isinstance(edir_val, str) else None,
        )

    # ================= 5) ACCESS EDGES ============================
    if access_edges:
        for e in access_edges:
            u, v = e["u_stop_id"], e["v_stop_id"]
            _set_node_pos_if_known(G, u, stops_df)
            _set_node_pos_if_known(G, v, stops_df)

            base_tt = float(e["travel_time_min_base"])

            G.add_edge(
                u, v,
                travel_time=base_tt,
                travel_time_min_base=base_tt,
                transfer_penalty_min=float(e.get("transfer_penalty_min", 0.0)),
                headway_min=0.0,
                distance_m=float(e.get("distance_m", 0.0)),
                mode=e.get("mode", "walk"),
                route_id=None,
                fare_id=e.get("fare_id", "F_WALK"),
            )

    return G
