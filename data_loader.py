import pandas as pd
import streamlit as st

EXCEL_PATH = "data/Multimodal_Transport_Dataset_v3.xlsx"

@st.cache_data(show_spinner=False)
def load_dataset(path: str = EXCEL_PATH):
    xls = pd.ExcelFile(path)
    stops        = pd.read_excel(xls, "stops")
    routes       = pd.read_excel(xls, "routes")
    edges        = pd.read_excel(xls, "edges")
    fares        = pd.read_excel(xls, "fares")
    fare_rules   = pd.read_excel(xls, "fare_rules")
    timetables   = pd.read_excel(xls, "timetables")
    traffic_rules = pd.read_excel(xls, "traffic_rules")
    
    # normalisasi kolom penting
    for df in (routes, edges):
        if "service_start" in df: df["service_start"] = pd.to_numeric(df["service_start"], errors="coerce")
        if "service_end"   in df: df["service_end"]   = pd.to_numeric(df["service_end"], errors="coerce")
        if "headway_min"   in df: df["headway_min"]   = pd.to_numeric(df["headway_min"], errors="coerce")

    return {
        "stops": stops,
        "routes": routes,
        "edges": edges,
        "fares": fares,
        "fare_rules": fare_rules,
        "timetables": timetables,
        "traffic_rules": traffic_rules,
    }
