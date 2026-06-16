import pandas as pd
import math

# ======================================================
# Penentu weekday/weekend
# ======================================================
def get_day_type(day_name):
    if day_name in ["Saturday", "Sunday"]:
        return "weekend"
    return "weekday"


# ======================================================
# Ambil multiplier dari traffic_rules_df
# ======================================================
def get_traffic_multiplier(current_minute, day_name, traffic_rules_df):
    if traffic_rules_df is None or traffic_rules_df.empty:
        return 1.0

    day_type = get_day_type(day_name)

    rules = traffic_rules_df[traffic_rules_df["day_type"] == day_type]
    if rules.empty:
        return 1.0

    for _, r in rules.iterrows():
        start = int(r["time_start"])
        end   = int(r["time_end"])
        if start <= current_minute <= end:
            return float(r["multiplier"])

    return 1.0  # default no congestion


# -------------------------------
# Cek jam operasi rute
# -------------------------------
def in_service_window(routes_df, route_id, current_minute):
    r = routes_df[routes_df["route_id"] == route_id]
    if r.empty:
        return True
    r = r.iloc[0]

    start = pd.to_numeric(r.get("service_start", 0), errors="coerce")
    end   = pd.to_numeric(r.get("service_end", 1440), errors="coerce")
    if math.isnan(start): start = 0
    if math.isnan(end):   end = 1440

    return (start <= current_minute <= end)


# -----------------------------------------
# Cek apakah KA aktif di hari ini
# -----------------------------------------
def is_train_operating(trainno, current_day):
    if not isinstance(trainno, str):
        return True

    weekend_trains = ["547F", "548F"]
    if trainno in weekend_trains:
        return current_day in ["Friday", "Saturday", "Sunday"]
    return True


# -------------------------------------------------
# Hitung waktu tunggu kereta/bus berikutnya
# -------------------------------------------------
def next_departure_wait_min(route_id, stop_id, timetables_df, current_minute, current_day,
                            routes_df=None, mode=None):
    # Jadwal kereta (prioritas)
    if timetables_df is not None and not timetables_df.empty:
        df = timetables_df[
            (timetables_df["route_id"] == route_id) &
            (timetables_df["stop_id"] == stop_id)
        ]
        if not df.empty:
            valid_rows = []
            for _, row in df.iterrows():
                trainno = str(row.get("trainno", ""))
                if is_train_operating(trainno, current_day):
                    valid_rows.append(row)

            if valid_rows:
                df = pd.DataFrame(valid_rows)
                dep_times = df["departure_minute"].dropna().astype(float).values
                dep_times = sorted(list(dep_times))
                if dep_times:
                    after = [t for t in dep_times if t >= current_minute]
                    if after:
                        return max(after[0] - current_minute, 0.1)
                    return (dep_times[0] + 1440) - current_minute

    # Headway bus
    if routes_df is not None and not routes_df.empty:
        r = routes_df[routes_df["route_id"] == route_id]
        if not r.empty:
            headway_val = pd.to_numeric(r.iloc[0].get("headway_min", None), errors="coerce")
            if pd.notna(headway_val) and headway_val > 0:
                return headway_val / 2.0

    # Default
    if mode == "bus":
        return 10.0
    elif mode in ["krl", "prameks", "railink"]:
        return 15.0
    return 10.0


# ----------------------------------------------------
# Ambil waktu kedatangan & keberangkatan kereta
# ----------------------------------------------------
def get_train_schedule_segment(route_id, timetables_df, current_day):
    if timetables_df is None or timetables_df.empty:
        return []

    df = timetables_df[timetables_df["route_id"] == route_id]
    if df.empty:
        return []

    df = df[df.apply(lambda r: is_train_operating(str(r.get("trainno","")), current_day), axis=1)]
    if df.empty:
        return []

    df = df.sort_values(by="stop_sequence")
    return df[["stop_id", "arrival_minute", "departure_minute", "trainno"]].to_dict("records")
