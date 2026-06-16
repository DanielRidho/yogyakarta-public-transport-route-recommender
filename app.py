import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import time
import time as pytime
import networkx as nx

from data_loader import load_dataset
from access_layer import generate_access_edges, generate_direct_walk_edge
from graph_builder import build_graph

# Engines baseline
from dijkstra_engine import (
    k_shortest_paths_by_time as dijkstra_paths,
    evaluate_paths as eval_dijkstra,
)
from astar_engine import (
    k_shortest_paths_by_time as astar_paths,
    evaluate_paths as eval_astar,
)
from bellman_engine import (
    k_shortest_paths_by_time as bellman_paths,
    evaluate_paths as eval_bellman,
)

# Engine fuzzy + traffic support
from dijkstra_fuzzy_engine import (
    k_shortest_paths_by_time as dijkstra_fz_paths,
    evaluate_paths as eval_dijkstra_fz,
)
# CONFIG
st.set_page_config(page_title="Hybrid Multimoda Routing", layout="wide")
st.title("🧭 Sistem Rekomendasi Rute Transportasi Multimoda")
st.caption("Yogyakarta • TJ • KRL • Prameks • Railink • YIA Ekspres")

USER_ORIGIN = "__USER_ORIGIN__"
USER_DEST   = "__USER_DEST__"

MAX_DISPLAY_CANDIDATES    = 6
ARRIVAL_SEARCH_WINDOW_MIN = 120
ARRIVAL_SEARCH_STEP_MIN   = 5

# LOAD DATA
data = load_dataset()

stops            = data["stops"]
routes_df        = data["routes"]
edges_df         = data["edges"]
fares_df         = data["fares"]
fare_rules_df    = data["fare_rules"]
timetables_df    = data["timetables"]
traffic_rules_df = data["traffic_rules"]

STOP_XY = {row["stop_id"]: (row["lat"], row["lon"]) for _, row in stops.iterrows()}

ROUTE_NAME = {
    str(r["route_id"]): str(r.get("route_name", str(r["route_id"])))
    for _, r in routes_df.iterrows()
}

ROUTE_COLOR = {
    r["route_id"]: r["mode_color"]
    for _, r in routes_df.dropna(subset=["mode_color"]).iterrows()
    if isinstance(r["mode_color"], str) and r["mode_color"].startswith("#")
}

MODE_COLOR_DEFAULT = {
    "walk": "#9aa0a6",
    "ojol": "#6fb1ff",
    "krl": "#1e40af",
    "prameks": "#3b82f6",
    "railink": "#7c3aed",
    "bus": "#ef4444",
}

# SESSION STATE INIT
for key, default in {
    "origin": None,
    "destination": None,
    "selection_mode": "Origin",
    "result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# HELPERS
def stop_name(stop_id):
    row = stops[stops["stop_id"] == stop_id]
    return row.iloc[0]["name"] if not row.empty else str(stop_id)

def stop_label(stop_id):
    row = stops[stops["stop_id"] == stop_id]
    if row.empty:
        return str(stop_id)
    return f"{stop_id} – {row.iloc[0]['name']}"

def hhmm(m):
    m = int(round(m))
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"

def seg_title(seg):
    mode = seg.get("mode", "").upper()
    rid  = seg.get("route_id")

    if mode in ("KRL", "PRAMEKS", "RAILINK") and rid:
        tt = timetables_df[timetables_df["route_id"] == rid]
        trainno = str(tt.iloc[0]["trainno"]) if (not tt.empty and "trainno" in tt.columns) else rid
        return f"{mode} • {trainno}"

    rname = ROUTE_NAME.get(str(rid), rid)
    return f"{mode} • {rname}"

def choose_best_and_candidates(cands, algo_is_fuzzy: bool):
    if not cands:
        return None, []
    if algo_is_fuzzy:
        cands_display = cands[:MAX_DISPLAY_CANDIDATES]
        best = sorted(
            cands_display,
            key=lambda r: (-r["fuzzy_score"], r["total_time"], r["total_cost"], r["transfer_count"])
        )[0]
    else:
        cands_display = cands[:1]
        best = sorted(
            cands_display,
            key=lambda r: (r["total_time"], r["total_cost"], r["transfer_count"])
        )[0]
    return best, cands_display

# ENGINE MAP
ENGINE_MAP = {
    "Dijkstra (Baseline)":      (dijkstra_paths,    eval_dijkstra,    False),
    "A* (Baseline)":            (astar_paths,       eval_astar,       False),
    "Bellman-Ford (Baseline)":  (bellman_paths,     eval_bellman,     False),
    "Dijkstra Fuzzy":           (dijkstra_fz_paths, eval_dijkstra_fz, True),
}

BASELINE_ALGOS = [
    "Dijkstra (Baseline)",
    "A* (Baseline)",
    "Bellman-Ford (Baseline)",
]

# SIDEBAR UI
with st.sidebar:
    st.header("⚙️ Panel Pengaturan")

    with st.expander("🚀 Pilih Algoritma", expanded=True):
        ALGO_OPTIONS = list(ENGINE_MAP.keys())
        alg_choice = st.selectbox("Algoritma", ALGO_OPTIONS)
        is_fuzzy = (alg_choice == "Dijkstra Fuzzy")

    with st.expander("🕒 Pengaturan Waktu", expanded=True):
        time_mode_label = st.radio(
            "Mode waktu",
            ["Berdasarkan waktu berangkat", "Berdasarkan waktu tiba"],
        )
        time_mode = "depart" if "berangkat" in time_mode_label.lower() else "arrival"

        day_name = st.selectbox("Hari perjalanan",
                                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

        if time_mode == "depart":
            t_depart = st.time_input("Jam berangkat", value=time(7,0))
            start_minute_input = t_depart.hour*60 + t_depart.minute
            arrival_minute_input = None
        else:
            t_arrive = st.time_input("Target jam tiba", value=time(8,0))
            arrival_minute_input = t_arrive.hour*60 + t_arrive.minute
            start_minute_input = None

    with st.expander("💳 Preferensi Pembayaran"):
        payment_pref = st.selectbox("Metode Pembayaran TransJogja", ["cashless","cash"])

    with st.expander("🅿️ Akses Titik → Halte", expanded=True):
        if is_fuzzy:
            top_n_access = st.slider("Jumlah halte akses terdekat", 1, 6, 3)
        else:
            top_n_access = 1
            st.caption("Baseline → otomatis 1 halte terdekat.")

    st.radio("Klik peta untuk memilih:", ["Origin","Destination"], key="selection_mode")

# INPUT MAP
m = folium.Map(location=[-7.79,110.37], zoom_start=12)

if st.session_state.origin:
    folium.Marker(st.session_state.origin, tooltip="Origin",
                  icon=folium.Icon(color="green")).add_to(m)

if st.session_state.destination:
    folium.Marker(st.session_state.destination, tooltip="Destination",
                  icon=folium.Icon(color="red")).add_to(m)

ret = st_folium(m, height=380, use_container_width=True)

if ret and ret.get("last_clicked"):
    lat, lon = ret["last_clicked"]["lat"], ret["last_clicked"]["lng"]
    if st.session_state.selection_mode == "Origin":
        st.session_state.origin = (lat,lon)
    else:
        st.session_state.destination = (lat,lon)

# RUNTIME BASELINE
def measure_baseline_runtimes(G, start_minute_used, day_name_used):
    stats = {}

    for alg in BASELINE_ALGOS:
        path_fn, eval_fn, _ = ENGINE_MAP[alg]
        t0 = pytime.perf_counter()

        try:
            alg_paths = path_fn(G, USER_ORIGIN, USER_DEST, k=MAX_DISPLAY_CANDIDATES)
        except Exception:
            alg_paths = []

        if not alg_paths:
            stats[alg] = {"runtime_ms":0,"best":None}
            continue

        cands = eval_fn(
            G, alg_paths, payment_pref,
            fares_df, fare_rules_df,
            stops, routes_df, timetables_df,
            start_minute_used, day_name_used
        )

        t1 = pytime.perf_counter()
        if not cands:
            stats[alg] = {"runtime_ms":(t1-t0)*1000,"best":None}
            continue

        best_base, _ = choose_best_and_candidates(cands, algo_is_fuzzy=False)
        stats[alg] = {
            "runtime_ms": (t1 - t0)*1000,
            "best": best_base,
        }

    return stats

# ROUTE COMPUTATION
def compute_route():
    path_fn, eval_fn, algo_is_fuzzy = ENGINE_MAP[alg_choice]

    if not (st.session_state.origin and st.session_state.destination):
        return

    olat,olon = st.session_state.origin
    dlat,dlon = st.session_state.destination

    # Access edges origin/destination
    acc_from = generate_access_edges(USER_ORIGIN, olat,olon, stops, top_n=top_n_access)
    acc_to   = generate_access_edges(USER_DEST,   dlat,dlon, stops, top_n=top_n_access)

    # reverse edges → DEST
    for e in list(acc_to):
        acc_to.append({
            "u_stop_id": e["v_stop_id"],
            "v_stop_id": USER_DEST,
            "mode": e["mode"],
            "distance_m": e.get("distance_m",0.0),
            "travel_time_min_base": e["travel_time_min_base"],
            "fare_id": e.get("fare_id","F_WALK"),
            "transfer_penalty_min": 0.0,
            "route_id": None,
        })

    direct_uv = generate_direct_walk_edge(USER_ORIGIN, olat,olon, USER_DEST, dlat,dlon)

    # Build graph
    G = build_graph(
        edges_df, routes_df, stops,
        timetables_df=timetables_df,
        access_edges=(acc_from + acc_to + direct_uv)
    )

    # path candidates
    try:
        paths = path_fn(G, USER_ORIGIN, USER_DEST, k=MAX_DISPLAY_CANDIDATES)
    except:
        st.error("🚫 Tidak ada rute yang menghubungkan titik asal dan tujuan.")
        st.session_state.result = None
        return

    if not paths:
        st.error("🚫 Tidak ditemukan jalur pada graf untuk titik tersebut.")
        st.session_state.result = None
        return

    # ===================== DEPART MODE =====================
    if time_mode == "depart":
        start_minute_used = start_minute_input

        cands = eval_fn(
            G, paths, payment_pref,
            fares_df, fare_rules_df,
            stops, routes_df, timetables_df,
            start_minute_used, day_name,

        )

        if not cands:
            st.error("🚫 Tidak ada layanan yang beroperasi pada jam tersebut.")
            st.session_state.result = None
            return

        best, cands_display = choose_best_and_candidates(cands, algo_is_fuzzy)
        actual_arrival = start_minute_used + best["total_time"]

    # ===================== ARRIVAL MODE =====================
    else:
        target_arrival = arrival_minute_input
        if target_arrival is None:
            st.error("⚠️ Waktu tiba belum diisi.")
            return

        dep_min = max(0, target_arrival - ARRIVAL_SEARCH_WINDOW_MIN)
        dep_max = target_arrival

        tested_runs = []

        for dep in range(dep_min, dep_max+1, ARRIVAL_SEARCH_STEP_MIN):
            cands = eval_fn(
                G, paths, payment_pref,
                fares_df, fare_rules_df,
                stops, routes_df, timetables_df,
                dep, day_name,
                traffic_rules_df=traffic_rules_df   # 🔥 DITAMBAHKAN
            )
            if not cands:
                continue

            best_loc, cands_loc = choose_best_and_candidates(cands, algo_is_fuzzy)
            arrival_loc = dep + best_loc["total_time"]

            tested_runs.append(
                dict(start=dep, arrival=arrival_loc,
                     best=best_loc, cands_display=cands_loc)
            )

        if not tested_runs:
            st.error("🚫 Tidak ada layanan relevan saat waktu tiba tersebut.")
            return

        before = [r for r in tested_runs if r["arrival"] <= target_arrival]

        def rank_key(run):
            gap = abs(target_arrival - run["arrival"])
            b = run["best"]
            if algo_is_fuzzy:
                return (gap, -b["fuzzy_score"], b["total_time"], b["total_cost"], b["transfer_count"])
            else:
                return (gap, b["total_time"], b["total_cost"], b["transfer_count"])

        chosen = sorted(before or tested_runs, key=rank_key)[0]

        start_minute_used = chosen["start"]
        actual_arrival    = chosen["arrival"]
        best              = chosen["best"]
        cands_display     = chosen["cands_display"]

    # runtime baseline
    runtime_stats = {}
    if alg_choice in BASELINE_ALGOS:
        runtime_stats = measure_baseline_runtimes(G, start_minute_used, day_name)

    # baseline comparator
    baseline_best_for_fuzzy = None
    if alg_choice == "Dijkstra Fuzzy":
        try:
            base_paths = dijkstra_paths(G, USER_ORIGIN, USER_DEST, k=1)
        except:
            base_paths = []
        if base_paths:
            base_cands = eval_dijkstra(
                G, base_paths, payment_pref,
                fares_df, fare_rules_df,
                stops, routes_df, timetables_df,
                start_minute_used, day_name
            )
            if base_cands:
                baseline_best_for_fuzzy,_ = choose_best_and_candidates(base_cands, False)

    st.session_state.result = dict(
        algo=alg_choice,
        best=best,
        cands=cands_display,
        time_mode=time_mode,
        start_minute=start_minute_used,
        actual_arrival=actual_arrival,
        target_arrival=arrival_minute_input if time_mode=="arrival" else None,
        runtime_stats=runtime_stats,
        baseline_best_for_fuzzy=baseline_best_for_fuzzy,
    )

# BUTTON
if st.button("⚡ Cari Rute", disabled=not(st.session_state.origin and st.session_state.destination)):
    with st.spinner("Menghitung rute..."):
        compute_route()

# OUTPUT
if not st.session_state.result:
    st.info("🗺️ Klik peta → pilih titik → tekan Cari Rute")
    st.stop()

result = st.session_state.result
best   = result["best"]
cands_display = result["cands"]
algo   = result["algo"]
is_fuzzy = (algo=="Dijkstra Fuzzy")
time_mode_used = result["time_mode"]
start_min_used = result["start_minute"]
actual_arrival = result["actual_arrival"]
runtime_stats = result.get("runtime_stats", {})
baseline_best_for_fuzzy = result.get("baseline_best_for_fuzzy")

# SUMMARY
st.subheader(f"📌 Hasil Terbaik — {algo}")

col1,col2,col3,col4 = st.columns(4)
col1.metric("Total waktu", f"{best['total_time']:.1f} mnt")
col2.metric("Total biaya", f"Rp {int(best['total_cost']):,}".replace(",","."))
col3.metric("Transfer",   best["transfer_count"])
col4.metric("Skor Fuzzy", best["fuzzy_score"] if is_fuzzy else "–")

dep_label = hhmm(start_min_used)
arr_label = hhmm(actual_arrival)

if time_mode_used=="depart":
    st.caption(f"Berangkat jam **{dep_label}** → tiba **{arr_label}**.")
else:
    target_label = hhmm(result["target_arrival"])
    st.caption(
        f"Target tiba: **{target_label}**. "
        f"Saran berangkat: **{dep_label}** → tiba **{arr_label}**."
    )

# MAP OUTPUT
olat,olon = st.session_state.origin
dlat,dlon = st.session_state.destination

mout = folium.Map(location=[(olat+dlat)/2,(olon+dlon)/2], zoom_start=13)

folium.Marker(st.session_state.origin, tooltip="Origin",
              icon=folium.Icon(color="green")).add_to(mout)
folium.Marker(st.session_state.destination, tooltip="Destination",
              icon=folium.Icon(color="red")).add_to(mout)

used_nodes = set()

for seg in best["segments"]:
    mode = seg["mode"]
    rid  = seg.get("route_id")
    base_color = MODE_COLOR_DEFAULT.get(mode,"#666666")
    color = ROUTE_COLOR.get(rid, base_color) if mode=="bus" else base_color

    coords=[]
    for u,v in seg["edges"]:
        if u==USER_ORIGIN:
            coords.append(st.session_state.origin)
        elif u==USER_DEST:
            coords.append(st.session_state.destination)
        else:
            coords.append(STOP_XY.get(u))

        used_nodes.add(u)

    last_v = seg["edges"][-1][1]
    coords.append(
        st.session_state.destination if last_v==USER_DEST else STOP_XY.get(last_v)
    )
    used_nodes.add(last_v)

    folium.PolyLine(coords, color=color, weight=6).add_to(mout)

for n in used_nodes:
    if n in STOP_XY:
        folium.CircleMarker(
            STOP_XY[n],
            radius=5, color="#111",
            fill=True,
            tooltip=stop_label(n)
        ).add_to(mout)

st.subheader("🗺️ Visualisasi Rute")
st_folium(mout, height=420, use_container_width=True)

# DETAIL TABLE
st.subheader("📋 Rincian Waktu Perjalanan")

detail_rows=[]
current = start_min_used

for i,seg in enumerate(best["segments"], start=1):
    wait    = seg.get("wait_min",0.0)
    ride    = seg.get("ride_min",0.0)
    penalty = seg.get("transfer_penalty_min",0.0)

    detail_rows.append({
        "Urutan":i,
        "Jenis": seg_title(seg),
        "Dari":  stop_label(seg["start_node"]),
        "Ke":    stop_label(seg["end_node"]),
        "Mulai Tunggu": hhmm(current),
        "Naik":          hhmm(current + wait),
        "Tiba":          hhmm(current + wait + ride),
        "Penalti (mnt)": penalty,
        "Total (mnt)": round(wait + ride + penalty,1),
        "Biaya (Rp)": int(seg.get("segment_cost",0)),
    })

    current += wait + ride + penalty

st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

# FUZZY CANDIDATES
if is_fuzzy:
    st.subheader("🧮 Kandidat Rute (Dijkstra Fuzzy)")
    dfc = pd.DataFrame([{
        "Rank": i+1,
        "Total Waktu (mnt)":r["total_time"],
        "Total Biaya (Rp)": r["total_cost"],
        "Transfer":         r["transfer_count"],
        "Skor Fuzzy":       r["fuzzy_score"],
    } for i,r in enumerate(cands_display)])
    st.dataframe(dfc, use_container_width=True)

# RUNTIME BASELINE
if runtime_stats:
    with st.expander("⏱️ Perbandingan Runtime Algoritma Baseline"):
        rows=[]
        for alg in BASELINE_ALGOS:
            s=runtime_stats.get(alg)
            if not s: continue
            b=s["best"]
            rows.append({
                "Algoritma": alg,
                "Runtime (ms)": round(s["runtime_ms"],2),
                "Waktu Terbaik (mnt)": b["total_time"] if b else "-",
                "Biaya Terbaik (Rp)":  b["total_cost"] if b else "-",
                "Transfer":            b["transfer_count"] if b else "-",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
