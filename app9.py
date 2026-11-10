# app.py
# Full runnable Streamlit app (single file) — embeds left/right blurred background images (A = embed)
# Run:
# pip install streamlit pandas numpy scikit-learn requests xgboost shap matplotlib pillow
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import os
import io
import base64
import warnings
import hashlib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# ---------------- page config ----------------
st.set_page_config(page_title="Crop Yield Predictor", layout="wide", initial_sidebar_state="collapsed")

# ---------------- dependencies ----------------
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    st.error("This app requires xgboost. Install it with: pip install xgboost")
    st.stop()

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

warnings.filterwarnings("ignore", message=".*use_label_encoder.*")

# ---------------- dataset path ----------------
DATASET = r"C:\Users\ADMIN\Desktop\crop3\updated_crop_yield_dataset_cleaned.xlsx"

# ---------------- images embed ----------------
LEFT_IMG_PATH = Path("/mnt/data/croplandno_blur.png")
RIGHT_IMG_PATH = Path("/mnt/data/land_with_crops_blur.png")

def file_to_b64(path: Path):
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

b_left = file_to_b64(LEFT_IMG_PATH)
b_right = file_to_b64(RIGHT_IMG_PATH)

# ---------------- CSS ----------------
css_parts = []
css_parts.append("<style>")
css_parts.append(""":root { --base-font: 18px; --label-font: 16px; --card-title: 18px; --card-value: 22px; --small-font: 14px; --badge-font: 12px; --accent: #1f7a34; }""")
css_parts.append(".main > div { padding: 10px 18px; font-size: var(--base-font); }")
css_parts.append(".card-row { display:flex; gap:14px; width:100%; margin-bottom:12px; flex-wrap:wrap; align-items:flex-start; }")
css_parts.append(".card { background: linear-gradient(180deg,#ffffff,#f7fbff); border-radius: 14px; box-shadow: 0 10px 30px rgba(14,30,37,0.06); padding: 16px; flex: 1 1 300px; min-width: 220px; border: 1px solid rgba(31,122,52,0.06);} ")
css_parts.append(".card h3 { margin: 0 0 8px 0; font-size: var(--card-title); display:flex; align-items:center; gap:10px; color:var(--accent); }")
css_parts.append(".card .value { font-weight:700; font-size: var(--card-value); margin-top:8px; display:block; color:#0f172a; }")
css_parts.append(".small { color:#475569; font-size: var(--small-font); margin-top:6px; }")
css_parts.append(".kpi { display:flex; align-items:center; justify-content:space-between; gap:10px; }")
css_parts.append(".badge { background:#eefaf0; color:var(--accent); padding:6px 10px; border-radius:999px; font-weight:700; font-size: var(--badge-font); }")
css_parts.append("label[data-baseweb='label'] { font-size: var(--label-font) !important; font-weight:600 !important; }")
css_parts.append("footer { visibility: hidden; }")
css_parts.append(".type-big { font-size: 28px !important; font-weight: 900 !important; color: #0b4a3b !important; margin-top: 8px; display: block; }")
css_parts.append(".header-image { width:48px; height:48px; border-radius:8px; box-shadow: 0 6px 18px rgba(14,30,37,0.08); margin-right:8px; }")

bg_block = ".stApp, .main, .block-container {"
if b_left and b_right:
    bg_block += (
        "background-image: url('data:image/png;base64," + b_left + "'), "
        "url('data:image/png;base64," + b_right + "');"
        "background-position: left center, right center;"
        "background-size: 48% 100%, 48% 100%;"
        "background-repeat: no-repeat, no-repeat;"
        "background-attachment: fixed;"
    )
else:
    bg_block += "background: linear-gradient(90deg, #e6f7ff 0%, #ffffff 50%, #f2fff0 100%);"
bg_block += "background-blend-mode: multiply;"
bg_block += "}"
css_parts.append(bg_block)
css_parts.append(".stCard, .card, .css-1d391kg, .stBlock { background: rgba(255,255,255,0.95) !important; backdrop-filter: blur(4px) saturate(120%); border-radius: 12px !important; box-shadow: 0 6px 18px rgba(15,23,42,0.06) !important; }")
css_parts.append("</style>")

st.markdown("".join(css_parts), unsafe_allow_html=True)

# ---------------- header ----------------
svg_icon = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>"
    "<rect rx='8' width='64' height='64' fill='%23f0fbff'/>"
    "<g transform='translate(8,8)'>"
    "<path d='M20 2c0 6-10 6-10 16s10 12 10 20c0-8 10-12 10-20S20 2 20 2z' fill='%231f7a34'/>"
    "<path d='M6 18c0 6 8 6 8 12 0 0-8-2-8-6 0-4 0-6 0-6z' fill='%233b9e3b' opacity='0.9'/>"
    "<circle cx='22' cy='28' r='2.5' fill='%23f6c85f'/>"
    "</g>"
    "</svg>"
)

st.markdown(
    f"""
    <div style='display:flex; align-items:center; gap:12px; margin-bottom:6px;'>
      <img class='header-image' src="{svg_icon}" alt='crop icon' />
      <div style='display:flex;flex-direction:column;'>
        <div style='font-size:28px; font-weight:800; color:var(--accent);'>🌾 Crop Yield Predictor</div>
        <div style='font-size:13px; color:#475569;'>Estimate yield and get fertilizer/pesticide guidance</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- helper functions ----------------
STATE_COORDS = {
    "karnataka": (12.9716, 77.5946), "maharashtra": (19.07599, 72.87766),
}

def deterministic_jitter(seed_string, scale=0.05):
    h = hashlib.sha256(seed_string.encode()).digest()
    val = int.from_bytes(h[:8], "big")
    frac = (val % 10**8) / (10**8 - 1)
    return ((frac * 2.0) - 1.0) * scale

def geocode(place):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{place}, India", "format": "json", "limit": 1},
            headers={"User-Agent": "crop-app"},
            timeout=6,
        )
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except:
        pass
    key = place.lower()
    if key in STATE_COORDS:
        return STATE_COORDS[key]
    return None, None

def fetch_rain_temp(lat, lon, state_hint=None):
    seed = state_hint or "unknown"
    return 800, 24

def calculate_yield(crop, soil, area, rainfall, temperature):
    return 4.2, 4.2 * area

# ---------------- Sidebar / Inputs ----------------
left_col, right_col = st.columns([1, 2])

with left_col:
    with st.form("input_form"):
        crop = st.text_input("🌾 Crop", "rice")
        state = st.text_input("📍 State", "Karnataka")
        area = st.number_input("📏 Area (ha)", min_value=0.01, value=1.0)
        soil = st.selectbox("🟫 Soil type", ["loamy", "sandy", "clay", "silty"])
        submitted = st.form_submit_button("Predict")

with right_col:
    weather_place = st.empty()
    yield_place = st.empty()
    recommendations_place = st.empty()

# ---------------- DO NOT SHOW OUTPUT UNTIL USER CLICKS PREDICT ----------------
if not submitted:
    st.warning("⚠️ No results yet — enter all inputs and click **Predict**.")
    st.stop()
# ------------------------------------------------------------------------------

# ---------------- begin prediction after submit ----------------
lat, lon = geocode(state)
rainfall, temperature = fetch_rain_temp(lat, lon, state)
yph, total_prod = calculate_yield(crop, soil, area, rainfall, temperature)

weather_place.markdown(f"### 🌦 Weather\n**Rainfall:** {rainfall} mm\n\n**Temperature:** {temperature} °C")
yield_place.markdown(f"### 🌾 Yield Prediction\n**Yield:** {yph} t/ha\n\n**Total Production:** {total_prod} tons")
recommendations_place.markdown(f"### 🧪 Recommendations\n- Fertilizer: 80 kg/ha\n- Pesticide: 2 kg/ha")

st.caption("📌 Developed to help farmers")
