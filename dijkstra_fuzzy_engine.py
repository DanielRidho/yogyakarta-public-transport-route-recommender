# Dijkstra Multi-Kriteria + Fuzzy + Traffic Jam

import math
import heapq
import itertools
from typing import Any, Dict, List, Tuple, FrozenSet, Optional

import networkx as nx
import pandas as pd

from schedule_utils import (
    in_service_window,
    next_departure_wait_min,
    get_traffic_multiplier,
)
from fare_utils import calculate_fare


# KONSTANTA WAKTU REAL
INTERMODAL_PENALTY   = 3.0
ROUTE_CHANGE_PENALTY = 2.0
TRANSFER_TAP_PENALTY = 1.0
DIRECT_ROUTE_BONUS   = -1.0

W_TIME     = 0.33
W_COST     = 0.33
W_TRANSFER = 0.34


# UTILITAS KECIL
def _corridor_from_route_id(rid: Any) -> Any:
    if isinstance(rid, str):
        return rid.split("_")[0]
    return rid


class Label:
    __slots__ = ("node", "time", "transfers", "corridors", "prev", "prev_edge")

    def __init__(self, node, time, transfers, corridors, prev, prev_edge):
        self.node = node
        self.time = time
        self.transfers = transfers
        self.corridors = corridors
        self.prev = prev
        self.prev_edge = prev_edge


def _dominates(a: Label, b: Label) -> bool:
    return (
        a.time <= b.time and
        a.transfers <= b.transfers and
        (a.time < b.time or a.transfers < b.transfers)
    )


def _is_dominated(candidate: Label, labels: List[Label]) -> bool:
    return any(_dominates(l, candidate) for l in labels)


# MC-DIJKSTRA
def _mc_dijkstra(G, source, target, max_labels_per_node=30):
    labels = {}
    heap = []
    counter = itertools.count()

    start = Label(source, 0.0, 0, frozenset(), None, None)
    labels[source] = [start]
    heapq.heappush(heap, (0.0, 0, next(counter), start))

    while heap:
        _, _, _, lab = heapq.heappop(heap)
        u = lab.node

        for _, v, data in G.out_edges(u, data=True):
            edge_time = float(data.get("travel_time",
                                       data.get("travel_time_min_base", 0.0)))

            mode = data.get("mode")
            rid = data.get("route_id")

            cid = _corridor_from_route_id(rid) if mode not in ("walk", "ojol") else None

            new_corr = lab.corridors | ({cid} if cid else set())
            new_trans = len(new_corr)
            new_time = lab.time + edge_time

            new_label = Label(v, new_time, new_trans, frozenset(new_corr),
                              lab, (u, v))

            cur = labels.get(v, [])

            if _is_dominated(new_label, cur):
                continue

            newlist = [lb for lb in cur if not _dominates(new_label, lb)]
            newlist.append(new_label)

            if len(newlist) > max_labels_per_node:
                newlist.sort(key=lambda x: (x.transfers, x.time))
                newlist = newlist[:max_labels_per_node]

            labels[v] = newlist
            heapq.heappush(heap, (new_time, new_trans, next(counter), new_label))

    return labels.get(target, [])


def _reconstruct_path(label: Label):
    out = []
    lab = label
    while lab:
        out.append(lab.node)
        lab = lab.prev
    return list(reversed(out))


def k_shortest_paths_by_time(G, source, target, k=6):
    labs = _mc_dijkstra(G, source, target)
    if not labs:
        return []
    labs_sorted = sorted(labs, key=lambda x: (x.transfers, x.time))
    return [_reconstruct_path(l) for l in labs_sorted[:k]]


# GROUP SEGMENT
def _group_segments(G, path):
    if len(path) < 2:
        return []
    segs = []
    edges = list(zip(path[:-1], path[1:]))

    i = 0
    while i < len(edges):
        u, v = edges[i]
        d = G[u][v]
        mode = d.get("mode")
        rid = d.get("route_id")

        block = [(u, v)]
        last = v
        j = i
        while j + 1 < len(edges):
            u2, v2 = edges[j + 1]
            d2 = G[u2][v2]
            if d2.get("mode") == mode and d2.get("route_id") == rid:
                block.append((u2, v2))
                last = v2
                j += 1
            else:
                break
        segs.append(
            dict(mode=mode, route_id=rid,
                 start_node=u, end_node=last, edges=block)
        )
        i = j + 1

    return segs


# WAKTU REAL + MACET
def compute_time(
    G,
    path,
    routes_df,
    timetables_df,
    stops_df,
    start_minute,
    current_day,
    traffic_rules_df=None,
):
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
        rid = seg.get("route_id")
        board = seg["start_node"]

        # Riding base
        ride_base = sum(
            float(G[u][v].get("travel_time_min_base", 0.0))
            for u, v in seg["edges"]
        )

        access = float(
            G[seg["edges"][0][0]][seg["edges"][0][1]]
            .get("transfer_penalty_min", 0.0)
        )

        # Cek jam operasi
        if rid and not in_service_window(routes_df, rid, int(current)):
            return math.inf, []

        # Waktu tunggu
        if mode in ("bus", "krl", "prameks", "railink"):
            wait = next_departure_wait_min(
                rid, board, timetables_df,
                int(current), current_day, routes_df, mode
            )
            if wait is None or math.isinf(wait):
                return math.inf, []
        else:
            wait = 0.0

        # Tambahkan MULTIPLIER KEMACETAN untuk BUS
        if mode == "bus" and traffic_rules_df is not None:
            multiplier = get_traffic_multiplier(
                int(current + wait), current_day, traffic_rules_df
            )
            ride = ride_base * float(multiplier)
        else:
            ride = ride_base

        # penalti perpindahan
        extra = 0.0
        if prev:
            if prev["mode"] != mode:
                extra += INTERMODAL_PENALTY
                if prev["mode"] == "bus" and not free_lookup.get(board, False):
                    extra += TRANSFER_TAP_PENALTY

            pc = _corridor_from_route_id(prev.get("route_id"))
            cc = _corridor_from_route_id(rid)
            if (
                prev["mode"] == "bus"
                and mode == "bus"
                and pc and cc
                and pc != cc
            ):
                extra += ROUTE_CHANGE_PENALTY
                if not free_lookup.get(board, False):
                    extra += TRANSFER_TAP_PENALTY

            if prev.get("route_id") == rid and rid is not None:
                extra += DIRECT_ROUTE_BONUS

        seg_time = access + wait + ride + extra
        total += seg_time
        current += seg_time

        details.append(
            dict(
                **seg,
                access_over_min=round(access, 2),
                wait_min=round(wait, 2),
                ride_min=round(ride, 2),     # sudah kena macet
                transfer_penalty_min=round(extra, 2),
                start_clock_min=current - seg_time,
                end_clock_min=current,
            )
        )
        prev = seg

    return round(total, 2), details


# FUZZY NORMALISASI
def _min_max_norm(vals):
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        return [0.5 for _ in vals]
    return [(v - mn) / (mx - mn) for v in vals]


def _tri(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 0
    return (c - x) / (c - b) if c != b else 0


def _m_time(x):
    return {
        "short":  _tri(x, 0, 0, 0.5),
        "medium": _tri(x, 0, 0.5, 1),
        "long":   _tri(x, 0.5, 1, 1)
    }


def _m_cost(x):
    return {
        "cheap":     _tri(x, 0, 0, 0.5),
        "medium":    _tri(x, 0, 0.5, 1),
        "expensive": _tri(x, 0.5, 1, 1)
    }


def _m_trans(x):
    return {
        "few":    _tri(x, 0, 0, 0.5),
        "medium": _tri(x, 0, 0.5, 1),
        "many":   _tri(x, 0.5, 1, 1)
    }


TIME_PREF = {"short": 1, "medium": 0.6, "long": 0.2}
COST_PREF = {"cheap": 1, "medium": 0.6, "expensive": 0.2}
TRANS_PREF = {"few": 1, "medium": 0.6, "many": 0.2}


def _fuzzy_eff(time_n, cost_n, trans_n):
    mu_t = _m_time(time_n)
    mu_c = _m_cost(cost_n)
    mu_x = _m_trans(trans_n)

    num = 0
    den = 0

    for t_lbl, mt in mu_t.items():
        if mt <= 0:
            continue
        for c_lbl, mc in mu_c.items():
            if mc <= 0:
                continue
            for x_lbl, mx in mu_x.items():
                if mx <= 0:
                    continue
                s = min(mt, mc, mx)
                w = (
                    W_TIME * TIME_PREF[t_lbl] +
                    W_COST * COST_PREF[c_lbl] +
                    W_TRANSFER * TRANS_PREF[x_lbl]
                )
                num += s * w
                den += s

    return 0.0 if den == 0 else (num / den) * 100.0


# EVALUASI + FUZZY + MACET
def evaluate_paths(
    G,
    paths,
    payment_pref,
    fares_df,
    fare_rules_df,
    stops_df,
    routes_df,
    timetables_df,
    start_minute,
    current_day,
    traffic_rules_df=None,
):
    raw = []
    for p in paths:
        t, segs = compute_time(
            G, p, routes_df, timetables_df, stops_df,
            start_minute, current_day,
            traffic_rules_df=traffic_rules_df,
        )
        if math.isinf(t):
            continue

        total_cost = 0
        prev = None
        corr = []

        for seg in segs:
            mode = seg["mode"]
            rid = seg.get("route_id")

            cost = calculate_fare(
                mode, rid, fares_df, fare_rules_df,
                payment_pref,
                u_stop=seg["start_node"], v_stop=seg["end_node"],
                current_day=current_day,
            )

            if prev and prev.get("route_id") == rid and prev.get("mode") == mode:
                cost = 0

            total_cost += cost
            seg["segment_cost"] = int(cost)

            if mode not in ("walk", "ojol") and rid:
                cid = _corridor_from_route_id(rid)
                corr.append(f"{mode}_{cid}")

            prev = seg

        raw.append({
            "path": p,
            "segments": segs,
            "total_time": float(t),
            "total_cost": int(total_cost),
            "transfer_count": int(len(set(corr))),
        })

    if not raw:
        return []

    # Normalisasi
    tn = _min_max_norm([x["total_time"] for x in raw])
    cn = _min_max_norm([x["total_cost"] for x in raw])
    xn = _min_max_norm([x["transfer_count"] for x in raw])

    out = []
    for r, t_n, c_n, x_n in zip(raw, tn, cn, xn):
        fs = _fuzzy_eff(t_n, c_n, x_n)
        r2 = r.copy()
        r2["fuzzy_score"] = round(fs, 2)
        out.append(r2)

    return sorted(
        out,
        key=lambda z: (-z["fuzzy_score"],
                       z["total_time"],
                       z["total_cost"],
                       z["transfer_count"])
    )
