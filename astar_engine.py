import math
import heapq
import itertools
import networkx as nx
import pandas as pd

from schedule_utils import (
    in_service_window,
    next_departure_wait_min,
    get_traffic_multiplier,
)
from fare_utils import calculate_fare

TRANSFER_TAP_PENALTY = 1.0
ROUTE_CHANGE_PENALTY = 2.0
INTERMODAL_PENALTY   = 3.0
DIRECT_ROUTE_BONUS   = -1.0

# =========================================================
# Haversine heuristic
# =========================================================
def _haversine(a, b):
    if ("lat" not in a) or ("lat" not in b):
        return 0.0
    R = 6371000.0
    dlat = math.radians(b["lat"] - a["lat"])
    dlon = math.radians(b["lon"] - a["lon"])
    x = (
        math.sin(dlat/2)**2
        + math.cos(math.radians(a["lat"]))
        * math.cos(math.radians(b["lat"]))
        * math.sin(dlon/2)**2
    )
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1-x)) / 80.0  # 4.8 km/h

# =========================================================
# A* pathfinder (single best path)
# =========================================================
def _astar_one(G, source, target):
    open_set = []
    heapq.heappush(open_set, (0, source, [source], 0))

    visited = {}

    while open_set:
        f, u, path, g = heapq.heappop(open_set)

        if (u in visited) and (visited[u] <= g):
            continue
        visited[u] = g

        if u == target:
            return path

        for _, v, data in G.out_edges(u, data=True):
            w = float(data.get("travel_time", 0.0))
            g2 = g + w
            h2 = _haversine(G.nodes[v], G.nodes[target])
            f2 = g2 + h2
            heapq.heappush(open_set, (f2, v, path + [v], g2))

    return None

# =========================================================
# Generate K routes by repeatedly penalizing used edges
# =========================================================
def k_shortest_paths_by_time(G, source, target, k=3):
    sols = []
    used_edge_penalty = {}

    for _ in range(max(6, k)):
        # mutate edge weights temporarily
        for (u, v) in used_edge_penalty:
            if G.has_edge(u, v):
                G[u][v]["_temp_added"] = used_edge_penalty[(u, v)]
            else:
                G[u][v]["_temp_added"] = 0.0

        # run A*
        for _, v, d in G.edges(data=True):
            d["travel_time_backup"] = d["travel_time"]
            d["travel_time"] = d["travel_time_backup"] + d.get("_temp_added",0)

        p = _astar_one(G, source, target)

        # restore weights
        for _, v, d in G.edges(data=True):
            d["travel_time"] = d["travel_time_backup"]

        if p is None:
            break

        sols.append(p)

        # penalize edges on this solution
        for u, v in zip(p[:-1], p[1:]):
            used_edge_penalty[(u, v)] = used_edge_penalty.get((u, v), 0.0) + 999

        if len(sols) >= k:
            break

    def path_cost(p):
        return sum(G[u][v].get("travel_time", 0.0) for u,v in zip(p[:-1],p[1:]))

    return sorted(sols, key=path_cost)[:k]

# =========================================================
# group segments
# =========================================================
def _group_segments(G, path):
    edges = list(zip(path[:-1], path[1:]))
    segs, i = [], 0
    while i < len(edges):
        u,v = edges[i]
        a = G[u][v]
        mode, rid = a.get("mode"), a.get("route_id")
        block = [(u,v)]
        j = i
        last = v
        while j+1 < len(edges):
            u2,v2 = edges[j+1]
            a2 = G[u2][v2]
            if a2.get("mode")==mode and a2.get("route_id")==rid:
                block.append((u2,v2))
                last = v2
                j+=1
            else:
                break
        segs.append(dict(mode=mode, route_id=rid,
                         start_node=u,end_node=last,edges=block))
        i = j+1
    return segs

# =========================================================
# compute_time
# =========================================================
def compute_time(G, path, routes_df, timetables_df, stops_df,
                 start_minute, current_day, traffic_rules_df=None):
    # EXACT COPY OF DIJKSTRA VERSION
    segs = _group_segments(G, path)
    current = float(start_minute)
    total = 0.0
    details = []
    prev = None

    free_lookup = {
        r["stop_id"]: bool(r.get("transfer_free_default", False))
        for _, r in stops_df.iterrows()
    }

    for seg in segs:
        mode = seg["mode"]
        rid  = seg.get("route_id")
        board = seg["start_node"]

        ride_base = sum(float(G[u][v].get("travel_time_min_base",0))
                        for u,v in seg["edges"])

        access = float(G[seg["edges"][0][0]][seg["edges"][0][1]]
                       .get("transfer_penalty_min",0))

        # jam operasi
        if rid and not in_service_window(routes_df, rid, int(current)):
            return math.inf, []

        # wait
        if mode in ("bus","krl","prameks","railink"):
            wait = next_departure_wait_min(
                route_id=rid,
                stop_id=board,
                timetables_df=timetables_df,
                current_minute=int(current),
                current_day=current_day,
                routes_df=routes_df,
                mode=mode,
            )
            if wait is None or math.isinf(wait):
                return math.inf,[]
        else:
            wait = 0.0

        # macet
        ride = ride_base
        if (traffic_rules_df is not None) and (mode=="bus"):
            m = get_traffic_multiplier(int(current+wait),
                                       current_day,
                                       traffic_rules_df)
            ride = ride_base * float(m)

        # penalti
        extra = 0.0
        if prev:
            if prev["mode"] != mode:
                extra += INTERMODAL_PENALTY
                if prev["mode"]=="bus" and not free_lookup.get(board,False):
                    extra += TRANSFER_TAP_PENALTY

            if (prev["mode"]=="bus" and mode=="bus"
                and seg.get("route_id")!=prev.get("route_id")):
                extra += ROUTE_CHANGE_PENALTY
                if not free_lookup.get(board,False):
                    extra += TRANSFER_TAP_PENALTY

            if prev.get("route_id")==rid and rid is not None:
                extra += DIRECT_ROUTE_BONUS

        seg_time = access + wait + ride + extra
        total += seg_time
        current += seg_time

        details.append(dict(**seg,
                            access_over_min=round(access,2),
                            wait_min=round(wait,2),
                            ride_min=round(ride,2),
                            transfer_penalty_min=round(extra,2),
                            start_clock_min=current-seg_time,
                            end_clock_min=current))

        prev = seg

    return round(total,2), details

# =========================================================
# evaluate_paths
# =========================================================
def evaluate_paths(G, paths, payment_pref, fares_df, fare_rules_df,
                   stops_df, routes_df, timetables_df, start_minute,
                   current_day, traffic_rules_df=None):
    results = []
    for p in paths:
        t, segs = compute_time(
            G, p, routes_df, timetables_df, stops_df,
            start_minute, current_day,
            traffic_rules_df=traffic_rules_df
        )
        if math.isinf(t):
            continue

        total_cost = 0.0
        prev = None
        active_routes = []

        for seg in segs:
            mode = seg["mode"]
            rid = seg.get("route_id")

            cost = calculate_fare(
                mode, rid, fares_df, fare_rules_df,
                payment_pref,
                u_stop=seg["start_node"],
                v_stop=seg["end_node"],
                current_day=current_day,
            )

            if prev and prev.get("route_id")==rid and prev.get("mode")==mode:
                cost = 0.0

            total_cost += cost
            seg["segment_cost"]=int(round(cost,0))

            if mode not in ("walk","ojol") and rid:
                key = f"{mode}_{rid}"
                if key not in active_routes:
                    active_routes.append(key)

            prev = seg

        transfers = len(active_routes)

        results.append(dict(
            path=p,
            segments=segs,
            total_time=t,
            total_cost=int(round(total_cost,0)),
            transfer_count=transfers,
        ))

    return sorted(results, key=lambda x:x["total_time"])
