import pandas as pd
import numpy as np

def calculate_fare(mode, route_id, fares_df, fare_rules_df, payment_type,
                   u_stop=None, v_stop=None, current_day="Monday"):
    if mode == "walk":
        return 0.0

    # Ambil baris tarif yang cocok
    fare_row = fares_df[
        (fares_df["mode"].str.lower() == mode.lower()) &
        (
            (fares_df["payment_type"].str.lower() == payment_type.lower())
            | (fares_df["payment_type"].str.lower() == "both")
        )
    ]
    if fare_row.empty:
        fare_row = fares_df[fares_df["mode"].str.lower() == mode.lower()]
    if fare_row.empty:
        return 0.0

    base_fare = float(fare_row.iloc[0].get("base_fare", 0))
    fare_id = fare_row.iloc[0]["fare_id"]
    rules = fare_rules_df[fare_rules_df["fare_id"] == fare_id]

    # Kasus khusus Railink (rule-based fare)
    if mode == "railink" and not rules.empty:
        rs = rules
        if u_stop and v_stop:
            rs2 = rs[
                ((rs["origin_stop_id"] == u_stop) & (rs["destination_stop_id"] == v_stop))
                | ((rs["origin_stop_id"] == v_stop) & (rs["destination_stop_id"] == u_stop))
            ]
            if not rs2.empty:
                fmin = float(rs2.iloc[0].get("fare_min", base_fare))
                fmax = float(rs2.iloc[0].get("fare_max", base_fare))
                val = rs2.iloc[0].get("fare_value", np.nan)
                return float(val) if pd.notna(val) else (fmin + fmax) / 2.0

        fmin = float(rs["fare_min"].min() or base_fare)
        fmax = float(rs["fare_max"].max() or base_fare)
        return (fmin + fmax) / 2.0

    # Kasus rule-based umum
    if not rules.empty:
        rs3 = rules[
            ((rules["origin_stop_id"] == u_stop) & (rules["destination_stop_id"] == v_stop))
            | ((rules["origin_stop_id"] == v_stop) & (rules["destination_stop_id"] == u_stop))
        ]
        if not rs3.empty:
            val = rs3.iloc[0].get("fare_value", np.nan)
            if pd.notna(val):
                return float(val)
            fmin = float(rs3.iloc[0].get("fare_min", base_fare))
            fmax = float(rs3.iloc[0].get("fare_max", base_fare))
            return (fmin + fmax) / 2.0

    # Default: flat fare
    return base_fare


# ==============================================================
# FUNGSI TAMBAHAN: Ambil batas maksimum tarif per moda
# ==============================================================

def get_max_fare_by_mode(mode, fares_df, fare_rules_df):
    if mode in ["railink"]:
        subset = fare_rules_df[
            fare_rules_df["fare_id"].str.contains(r"F_YA(_EX)?", case=False, na=False)
        ]
        if not subset.empty:
            return subset["fare_value"].max()
        return 40000

    elif mode in ["krl", "prameks"]:
        subset = fares_df[fares_df["mode"].str.lower() == mode.lower()]
        if not subset.empty:
            return subset["base_fare"].max()
        return 8000

    elif mode == "bus":
        subset = fares_df[fares_df["mode"].str.lower() == "bus"]
        if not subset.empty:
            return subset["base_fare"].max()
        return 3500

    elif mode == "walk":
        return 0.0

    # Default fallback untuk moda lain
    return 15000


# ==============================================================
# Normalisasi biaya ke domain fuzzy (0–15000)
# ==============================================================

def normalize_cost_for_fuzzy(mode, total_cost, fares_df, fare_rules_df):
    if mode == "walk":
        return 0.0

    max_fare = get_max_fare_by_mode(mode, fares_df, fare_rules_df)
    if max_fare <= 0:
        max_fare = 15000  # fallback jika data tidak ada

    # Konversi proporsional
    normalized_cost = (total_cost / max_fare) * 15000
    normalized_cost = float(np.clip(normalized_cost, 0, 15000))
    return normalized_cost
