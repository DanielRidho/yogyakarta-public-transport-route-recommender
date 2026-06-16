# app2.py — Viewer Jaringan Rute & Halte Multimoda
# -------------------------------------------------
# Fitur:
# - Peta semua rute (TransJogja, KRL, Prameks, KA Bandara)
# - Filter:
#     • Moda   : Semua / Bus / KRL / Prameks / Railink
#     • Rute   : Semua rute / salah satu route_name (1A, 2A, 5B, ...)
#     • Arah   : hanya muncul jika rute spesifik dipilih
# - Pilihan tampilan:
#     • Hanya halte
#     • Hanya jalur (polyline)
#     • Halte + jalur
#     • Sembunyikan semua
# - Tooltip/popup:
#     • Halte : "STOP_ID – Nama Halte"
#     • Jalur : "MODE • NAMA_RUTE (Arah: HalteAwal → HalteAkhir)"
# -------------------------------------------------

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from data_loader import load_dataset


# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================
st.set_page_config(page_title="Viewer Jaringan Multimoda", layout="wide")
st.title("🗺️ Viewer Jaringan Rute Transportasi Multimoda Yogyakarta")
st.caption("TransJogja • KRL Jogja–Solo • Prameks • KA Bandara YIA (Railink)")


# =========================================================
# 2. LOAD DATASET
# =========================================================
data = load_dataset()
stops_df     = data["stops"]
routes_df    = data["routes"]
edges_df_all = data["edges"]

# Hanya pakai edge publik (tanpa walk/ojol) di viewer ini
PUBLIC_MODES = {"bus", "krl", "prameks", "railink"}
edges_df = edges_df_all[edges_df_all["mode"].isin(PUBLIC_MODES)].copy()

# Posisi halte
STOP_XY = {
    row["stop_id"]: (float(row["lat"]), float(row["lon"]))
    for _, row in stops_df.dropna(subset=["stop_id", "lat", "lon"]).iterrows()
}

# Info rute per route_id (1 baris = 1 kombinasi route_name + direction)
ROUTE_INFO = {}
for _, r in routes_df.iterrows():
    rid = r.get("route_id")
    if pd.isna(rid):
        continue
    rid = str(rid)
    ROUTE_INFO[rid] = {
        "route_name": r.get("route_name", rid),
        "mode": str(r.get("mode", "")).lower(),
        "direction": r.get("direction"),
        "mode_color": r.get("mode_color"),
    }

# Warna default per mode (fallback jika mode_color kosong)
MODE_COLOR_DEFAULT = {
    "bus": "#ef4444",       # merah
    "krl": "#1e40af",       # biru tua
    "prameks": "#3b82f6",   # biru
    "railink": "#7c3aed",   # ungu
}

# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================
def stop_name(stop_id):
    """Ambil nama halte/stasiun dari stops_df."""
    row = stops_df[stops_df["stop_id"] == stop_id]
    if row.empty:
        return str(stop_id)
    return str(row.iloc[0]["name"])


def stop_label(stop_id):
    """Label gabungan untuk tooltip/popup halte."""
    return f"{stop_id} – {stop_name(stop_id)}"


def compute_route_endpoints(route_id: str):
    """
    Cari halte awal & akhir dari suatu route_id.
    Dipakai untuk label arah: 'HalteAwal → HalteAkhir'.
    """
    sub = edges_df[edges_df["route_id"] == route_id][["u_stop_id", "v_stop_id"]].dropna()
    if sub.empty:
        return None, None

    # Degre berdasarkan graf tidak berarah (untuk mendeteksi ujung koridor)
    deg = {}
    for _, row in sub.iterrows():
        u = row["u_stop_id"]
        v = row["v_stop_id"]
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1

    endpoints = [sid for sid, d in deg.items() if d == 1]

    if len(endpoints) >= 2:
        start_id, end_id = endpoints[0], endpoints[1]
    else:
        # fallback: pakai edge pertama & terakhir
        first = sub.iloc[0]
        last = sub.iloc[-1]
        start_id = first["u_stop_id"]
        end_id   = last["v_stop_id"]

    return start_id, end_id


def build_direction_labels():
    """
    Bangun label arah per (route_name, direction):
        (1A, A) -> 'HalteAwal → HalteAkhir'
    """
    labels = {}
    for _, r in routes_df.iterrows():
        rid = r.get("route_id")
        if pd.isna(rid):
            continue
        rid = str(rid)
        route_name = r.get("route_name", rid)
        direction  = r.get("direction")
        if pd.isna(route_name) or pd.isna(direction):
            continue

        start_id, end_id = compute_route_endpoints(rid)
        if start_id is None or end_id is None:
            continue

        labels[(str(route_name), str(direction))] = f"{stop_name(start_id)} → {stop_name(end_id)}"
    return labels


ROUTE_DIR_LABEL = build_direction_labels()


def get_color_for_route_id(route_id: str) -> str:
    """Tentukan warna jalur berdasarkan mode_color atau default mode."""
    info = ROUTE_INFO.get(route_id, {})
    color = info.get("mode_color")
    if isinstance(color, str) and color.startswith("#") and len(color) in (4, 7):
        return color
    mode = info.get("mode", "")
    return MODE_COLOR_DEFAULT.get(mode, "#666666")


def get_mode_display_name(mode_key: str) -> str:
    """Label cantik untuk dropdown moda."""
    mapping = {
        "all": "Semua Moda",
        "bus": "Bus TransJogja",
        "krl": "KRL Jogja–Solo",
        "prameks": "Prambanan Ekspres",
        "railink": "KA Bandara YIA (Railink)",
    }
    return mapping.get(mode_key, mode_key)


# =========================================================
# 4. SIDEBAR FILTER
# =========================================================
st.sidebar.header("⚙️ Filter Visualisasi")

# 4.1 Pilih moda
mode_options = [
    ("all", get_mode_display_name("all")),
    ("bus", get_mode_display_name("bus")),
    ("krl", get_mode_display_name("krl")),
    ("prameks", get_mode_display_name("prameks")),
    ("railink", get_mode_display_name("railink")),
]
mode_display_list = [label for _, label in mode_options]
mode_choice_label = st.sidebar.selectbox("Moda", mode_display_list, index=0)
mode_choice_key = [key for key, label in mode_options if label == mode_choice_label][0]

# Filter routes berdasarkan moda terpilih
if mode_choice_key == "all":
    routes_filtered = routes_df.copy()
else:
    routes_filtered = routes_df[routes_df["mode"].str.lower() == mode_choice_key].copy()

# 4.2 Pilih rute (berdasarkan route_name, bukan route_id)
route_names = sorted(
    {str(rn) for rn in routes_filtered["route_name"].dropna().unique()}
)
route_select_options = ["Semua Rute"] + route_names
route_choice = st.sidebar.selectbox("Rute", route_select_options, index=0)

# 4.3 Pilih arah (hanya muncul jika rute spesifik dipilih)
direction_choice = None
direction_label_to_value = {}
if route_choice != "Semua Rute":
    # Ambil baris-baris untuk route_name tersebut
    rsub = routes_filtered[routes_filtered["route_name"] == route_choice]
    dir_items = []
    for _, r in rsub.iterrows():
        direction = str(r.get("direction", "")).strip()
        if not direction:
            continue
        label_dir = ROUTE_DIR_LABEL.get((route_choice, direction))
        if label_dir:
            label = f"Arah {direction}: {label_dir}"
        else:
            label = f"Arah {direction}"
        dir_items.append((label, direction))

    if dir_items:
        dir_labels = [lab for lab, _ in dir_items]
        default_idx = 0
        selected_label = st.sidebar.selectbox("Arah", dir_labels, index=default_idx)
        # mapping label -> direction
        direction_choice = dict(dir_items)[selected_label]

# 4.4 Pilihan elemen yang ditampilkan
show_option = st.sidebar.radio(
    "Elemen yang ditampilkan",
    ["Halte + Jalur", "Hanya Jalur", "Hanya Halte", "Sembunyikan semua"],
    index=0,
    help="Pilih kombinasi tampilan antara jalur rute dan titik halte.",
)


# =========================================================
# 5. FILTER DATA BERDASARKAN PILIHAN USER
# =========================================================
edges_view = edges_df.copy()

# Filter moda
if mode_choice_key != "all":
    edges_view = edges_view[edges_view["mode"].str.lower() == mode_choice_key]

# Filter rute (route_name) & arah
if route_choice != "Semua Rute":
    # Cari semua route_id yang punya route_name itu (dan cocok dengan moda jika difilter)
    rsub = routes_filtered[routes_filtered["route_name"] == route_choice]
    if direction_choice is not None:
        rsub = rsub[rsub["direction"].astype(str) == str(direction_choice)]
    route_ids_chosen = {str(rid) for rid in rsub["route_id"].dropna().unique()}
    if route_ids_chosen:
        edges_view = edges_view[edges_view["route_id"].astype(str).isin(route_ids_chosen)]
    else:
        edges_view = edges_view.iloc[0:0]  # kosong


# Tentukan set halte yang terpakai oleh edges_view
used_stop_ids = set()
for _, e in edges_view.iterrows():
    used_stop_ids.add(e["u_stop_id"])
    used_stop_ids.add(e["v_stop_id"])

# =========================================================
# 6. BANGUN PETA FOLIUM
# =========================================================
# Tentukan pusat peta (mean dari semua halte atau fallback Yogyakarta)
if STOP_XY:
    lat_mean = sum(lat for lat, _ in STOP_XY.values()) / len(STOP_XY)
    lon_mean = sum(lon for _, lon in STOP_XY.values()) / len(STOP_XY)
else:
    lat_mean, lon_mean = -7.79, 110.37

m = folium.Map(location=[lat_mean, lon_mean], zoom_start=12)

# 6.1 Gambar jalur (polyline) jika diminta
if show_option in ("Halte + Jalur", "Hanya Jalur") and not edges_view.empty:
    # Kelompokkan berdasarkan route_id agar warna/tooltipnya konsisten
    for rid, group in edges_view.groupby("route_id"):
        rid = str(rid)
        info = ROUTE_INFO.get(rid, {})
        mode = info.get("mode", "")
        route_name = info.get("route_name", rid)
        direction = info.get("direction", "")
        arah_label = ROUTE_DIR_LABEL.get((str(route_name), str(direction)), "")

        color = get_color_for_route_id(rid)

        # Tooltip untuk jalur
        if arah_label:
            tooltip_text = f"{mode.upper()} • {route_name} ({arah_label})"
        else:
            tooltip_text = f"{mode.upper()} • {route_name}"

        # Gambar tiap edge sebagai segmen polyline
        for _, e in group.iterrows():
            u = e["u_stop_id"]
            v = e["v_stop_id"]
            if u not in STOP_XY or v not in STOP_XY:
                continue
            coords = [STOP_XY[u], STOP_XY[v]]
            folium.PolyLine(
                coords,
                color=color,
                weight=5,
                opacity=0.9,
                tooltip=tooltip_text,
            ).add_to(m)

# 6.2 Gambar halte jika diminta
if show_option in ("Halte + Jalur", "Hanya Halte"):
    # Jika ada filter rute, tampilkan halte yang kepakai saja; kalau tidak, semua halte
    if route_choice == "Semua Rute" and mode_choice_key == "all":
        stops_to_draw = STOP_XY.keys()
    else:
        stops_to_draw = used_stop_ids

    for sid in stops_to_draw:
        if sid not in STOP_XY:
            continue
        folium.CircleMarker(
            location=STOP_XY[sid],
            radius=4,
            color="#111111",
            fill=True,
            fill_opacity=0.9,
            tooltip=stop_label(sid),
            popup=stop_label(sid),
        ).add_to(m)

st.subheader("🗺️ Visualisasi Jaringan Rute & Halte")
st_folium(m, height=500, use_container_width=True)

# Info kecil kalau tidak ada data
if edges_view.empty and show_option in ("Halte + Jalur", "Hanya Jalur"):
    st.warning("Tidak ada jalur yang cocok dengan kombinasi filter saat ini.")
