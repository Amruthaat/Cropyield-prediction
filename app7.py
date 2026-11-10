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

# ---------------- dataset path (unused for images) ----------------
DATASET = r"C:\Users\ADMIN\Desktop\crop3\updated_crop_yield_dataset_cleaned.xlsx"  # change if needed

# ---------------- images to embed (Option A) ----------------
LEFT_IMG_PATH = Path("/mnt/data/croplandno_blur.png")
RIGHT_IMG_PATH = Path("/mnt/data/land_with_crops_blur.png")

def file_to_b64(path: Path):
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

b_left = file_to_b64(LEFT_IMG_PATH)
b_right = file_to_b64(RIGHT_IMG_PATH)

# ---------------- inject CSS for styling ----------------
css_parts = []
css_parts.append("<style>")
# changed accent from blue to dark green (#1f7a34)
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
# big type for fertilizer/pesticide
css_parts.append(".type-big { font-size: 28px !important; font-weight: 900 !important; color: #0b4a3b !important; margin-top: 8px; display: block; }")
css_parts.append(".header-image { width:48px; height:48px; border-radius:8px; box-shadow: 0 6px 18px rgba(14,30,37,0.08); margin-right:8px; }")
# background
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
css_parts.append("header, footer { background: transparent !important; }")
css_parts.append("</style>")

st.markdown("".join(css_parts), unsafe_allow_html=True)

# ---------------- small inline crop-yield SVG (embedded in header) ----------------
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

# ---------------- header ----------------
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

# ---------------- deterministic helper & meteorology ----------------
STATE_COORDS = {
    "karnataka": (12.9716, 77.5946), "maharashtra": (19.07599, 72.87766), "tamil nadu": (13.0827, 80.2707),
    "kerala": (9.9312, 76.2673), "uttar pradesh": (26.8467, 80.9462), "bihar": (25.5941, 85.1376),
    "west bengal": (22.5726, 88.3639), "andhra pradesh": (16.5062, 80.6480), "telangana": (17.3850, 78.4867),
    "gujarat": (23.0225, 72.5714), "rajasthan": (26.9124, 75.7873), "punjab": (31.1471, 75.3412),
    "haryana": (29.0588, 76.0856), "odisha": (20.2961, 85.8245), "assam": (26.1445, 91.7362),
    "delhi": (28.6448, 77.2167), "madhya pradesh": (23.2599, 77.4126), "chhattisgarh": (21.1458, 81.3858),
    "jharkhand": (23.3441, 85.3096), "goa": (15.2993, 74.1240), "manipur": (24.8170, 93.9368),
    "mizoram": (23.1645, 92.9376), "nagaland": (26.1584, 94.5624), "tripura": (23.9408, 91.9882),
    "sikkim": (27.3333, 88.6167), "uttarakhand": (30.0668, 79.0193), "ladakh": (34.1526, 77.5770),
}

def deterministic_jitter(seed_string, scale=0.05):
    h = hashlib.sha256(seed_string.encode("utf-8")).digest()
    val = int.from_bytes(h[:8], "big")
    frac = (val % 10**8) / (10**8 - 1)
    mapped = (frac * 2.0) - 1.0
    return mapped * scale

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
    except Exception:
        pass
    key = str(place).strip().lower()
    if key in STATE_COORDS:
        return STATE_COORDS[key]
    for k in STATE_COORDS:
        if k in key or key in k:
            return STATE_COORDS[k]
    return None, None

def fetch_rain_temp(lat, lon, state_hint=None):
    try:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=365)
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "precipitation_sum,temperature_2m_mean",
                "timezone": "Asia/Kolkata",
            },
            timeout=10,
        )
        d = r.json().get("daily", {})
        rain = d.get("precipitation_sum", [])
        temp = d.get("temperature_2m_mean", [])
        if rain and temp:
            rain_sum = float(np.sum(rain))
            temp_mean = float(np.nanmean(temp))
            seed = state_hint if state_hint else f"{lat},{lon}"
            jitter = deterministic_jitter(seed, scale=0.05)
            rain_sum = max(0.0, rain_sum * (1.0 + jitter))
            temp_mean = temp_mean * (1.0 + jitter * 0.6)
            return round(rain_sum, 2), round(temp_mean, 2)
    except Exception:
        pass

    seed = state_hint if state_hint else (f"{lat},{lon}" if lat is not None else "unknown")
    base_rain = 800.0; base_temp = 24.0
    if state_hint:
        k = state_hint.strip().lower()
        if any(s in k for s in ["kerala", "karnataka", "goa", "maharashtra", "tamil"]):
            base_rain = 1200.0; base_temp = 26.0
        elif any(s in k for s in ["rajasthan", "punjab", "haryana", "gujarat"]):
            base_rain = 400.0; base_temp = 25.0
        elif any(s in k for s in ["uttar pradesh", "bihar", "west bengal"]):
            base_rain = 900.0; base_temp = 24.0
        elif any(s in k for s in ["assam", "meghalaya", "northeast", "manipur", "mizoram"]):
            base_rain = 2000.0; base_temp = 23.0
        elif "ladakh" in k or "jammu" in k or "kashmir" in k:
            base_rain = 200.0; base_temp = 8.0

    jitter = deterministic_jitter(seed, scale=0.25)
    rain_sum = max(0.0, base_rain * (1.0 + jitter))
    temp_mean = base_temp * (1.0 + jitter * 0.2)
    return round(rain_sum, 2), round(temp_mean, 2)

# ---------------- yield estimator ----------------
def calculate_yield(crop, soil, area, rainfall, temperature):
    base = {"rice": 4, "wheat": 3, "maize": 5}.get(str(crop).lower(), 2)
    if rainfall is None: rainfall = 800
    if temperature is None: temperature = 24
    crop_l = str(crop).lower()
    if crop_l == "rice": ideal = (1000,2000)
    elif crop_l == "wheat": ideal = (300,800)
    else: ideal = (400,1200)
    if rainfall < ideal[0]:
        rf = max(0.5, 1 - (ideal[0] - rainfall) / ideal[0])
    elif rainfall > ideal[1]:
        rf = max(0.7, 1 - (rainfall - ideal[1]) / (ideal[1] + 2000))
    else:
        rf = 1.0
    tf = max(0.6, 1 - 0.03 * abs(temperature - 24))
    soil_factor = {"loamy":1.05,"sandy":0.85,"clay":0.95,"silty":1.02}.get(str(soil).lower(),1.0)
    yph = base * rf * tf * soil_factor
    yph = max(yph, 0.2)
    return round(yph,4), round(yph*area,4)

# ---------------- feature encoder helpers ----------------
def fit_feature_encoders(df, feature_cols):
    enc = {}; numeric_means = {}
    for c in feature_cols:
        if c not in df.columns: continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_means[c] = float(df[c].mean()) if df[c].notna().any() else 0.0
        else:
            le = LabelEncoder(); le.fit(df[c].astype(str).fillna("___MISSING___")); enc[c] = le
    return enc, numeric_means

def transform_features(df, feature_cols, enc_dict, numeric_means):
    out = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c not in df.columns: df[c] = np.nan
        if c in enc_dict:
            values = df[c].astype(str).fillna("___MISSING___"); le = enc_dict[c]; known = set(le.classes_)
            transformed = [int(le.transform([v])[0]) if v in known else -1 for v in values]
            out[c] = transformed
        else:
            if pd.api.types.is_numeric_dtype(df[c]) or c in numeric_means:
                mean_val = numeric_means.get(c, float(df[c].mean()) if df[c].notna().any() else 0.0)
                out[c] = df[c].astype(float).fillna(mean_val)
            else:
                out[c] = 0.0
    return out[feature_cols].astype(float)

# ---------------- model placeholders ----------------
fert_type_model = fert_amt_model = pest_type_model = pest_amt_model = None
fert_type_knn = pest_type_knn = None
fert_type_label_enc = pest_type_label_enc = None
feature_encoders = {}; feature_means = {}
crop_options = None; state_options = None; feature_cols = []
most_freq_fert_label = None; most_freq_pest_label = None

# ---------------- load dataset & train if available ----------------
if os.path.exists(DATASET):
    try:
        df = pd.read_excel(DATASET) if DATASET.lower().endswith(".xlsx") else pd.read_csv(DATASET)
        cols = df.columns.tolist()
        def find_col(cands):
            for cand in cands:
                for c in cols:
                    if cand.lower() in c.lower(): return c
            return None

        crop_col = find_col(["crop","crop_name","cropname"])
        state_col = find_col(["state","district","region"])
        soil_col = find_col(["soil","soil_type"])
        area_col = find_col(["area","area_ha"])
        fertType_col = find_col(["fertilizer_type","fert_type","fertilizername"])
        fertAmt_col = find_col(["fertilizer_kg","fertilizer_amount","fert_amt"])
        pestType_col = find_col(["pesticide_type","pest_type","pesticide","pest"])
        pestAmt_col = find_col(["pesticide_amount","pesticide_kg","pest_amt"])

        feature_cols = [c for c in [crop_col,state_col,soil_col,area_col] if c]

        if crop_col: crop_options = sorted(df[crop_col].dropna().astype(str).unique().tolist())
        if state_col: state_options = sorted(df[state_col].dropna().astype(str).unique().tolist())

        if feature_cols:
            feature_cols = [c for c in feature_cols if c in df.columns]
            feature_encoders, feature_means = fit_feature_encoders(df[feature_cols], feature_cols)

        if 'fertType_col' in locals() and fertType_col and df[fertType_col].notna().sum()>10 and feature_cols:
            d = df.dropna(subset=[fertType_col]); X = d[feature_cols].copy(); y = d[fertType_col].astype(str).fillna("___MISSING___")
            most_freq_fert_label = y.value_counts().index[0] if not y.empty else None
            if y.nunique() > 1:
                Xenc = transform_features(X, feature_cols, feature_encoders, feature_means)
                fert_type_label_enc = LabelEncoder(); y_enc = fert_type_label_enc.fit_transform(y)
                fert_type_model = XGBClassifier(n_estimators=200, eval_metric="mlogloss", verbosity=0); fert_type_model.fit(Xenc, y_enc)
                try: fert_type_knn = KNeighborsClassifier(n_neighbors=3); fert_type_knn.fit(Xenc, y_enc)
                except Exception: fert_type_knn = None

        if 'fertAmt_col' in locals() and fertAmt_col and df[fertAmt_col].notna().sum()>20 and feature_cols:
            d = df.dropna(subset=[fertAmt_col]); X = d[feature_cols].copy(); y = d[fertAmt_col].astype(float).copy()
            Xenc = transform_features(X, feature_cols, feature_encoders, feature_means)
            fert_amt_model = XGBRegressor(n_estimators=200, verbosity=0); fert_amt_model.fit(Xenc, y, eval_metric="rmse")

        if 'pestType_col' in locals() and pestType_col and df[pestType_col].notna().sum()>10 and feature_cols:
            d = df.dropna(subset=[pestType_col]); X = d[feature_cols].copy(); y = d[pestType_col].astype(str).fillna("___MISSING___")
            most_freq_pest_label = y.value_counts().index[0] if not y.empty else None
            if y.nunique() > 1:
                Xenc = transform_features(X, feature_cols, feature_encoders, feature_means)
                pest_type_label_enc = LabelEncoder(); y_enc = pest_type_label_enc.fit_transform(y)
                pest_type_model = XGBClassifier(n_estimators=200, eval_metric="mlogloss", verbosity=0); pest_type_model.fit(Xenc, y_enc)
                try: pest_type_knn = KNeighborsClassifier(n_neighbors=3); pest_type_knn.fit(Xenc, y_enc)
                except Exception: pest_type_knn = None

        if 'pestAmt_col' in locals() and pestAmt_col and df[pestAmt_col].notna().sum()>20 and feature_cols:
            d = df.dropna(subset=[pestAmt_col]); X = d[feature_cols].copy(); y = d[pestAmt_col].astype(float).copy()
            Xenc = transform_features(X, feature_cols, feature_encoders, feature_means)
            pest_amt_model = XGBRegressor(n_estimators=200, verbosity=0); pest_amt_model.fit(Xenc, y, eval_metric="rmse")

    except Exception:
        st.warning("Dataset load / training encountered an issue; defaults will be used where necessary.")

# ---------------- layout ----------------
left_col, right_col = st.columns([1.0, 1.8], gap="small")

show_debug = False

with left_col:
    with st.form("input_form"):
        if crop_options:
            crop = st.selectbox("🌾  Crop", crop_options, key="crop_input")
        else:
            crop = st.text_input("🌾  Crop", "rice", key="crop_input")

        if state_options:
            state = st.selectbox("📍  State ", state_options, index=0, key="state_input")
        else:
            state = st.text_input("📍  State ", "Karnataka", key="state_input")

        area = st.number_input("📏  Area (ha)", min_value=0.01, value=1.0, step=0.1, key="area_input")
        soil = st.selectbox("🟫  Soil type", ["loamy", "sandy", "clay", "silty", "peaty", "chalky"], key="soil_input")

        show_debug = st.checkbox("🛠️ ", value=False, key="debug_input")
        submitted = st.form_submit_button("Predict")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

with right_col:
    weather_place = st.empty()
    yield_place = st.empty()
    recommendations_place = st.empty()

if submitted:
    lat, lon = geocode(state)
    seed_hint = f"{state}|{crop}|{soil}|{area}"
    if lat is not None and lon is not None:
        rainfall, temperature = fetch_rain_temp(lat, lon, state_hint=seed_hint)
    else:
        rainfall, temperature = fetch_rain_temp(None, None, state_hint=seed_hint)

    yph, total_prod = calculate_yield(crop, soil, area, rainfall, temperature)

    if feature_cols:
        try:
            row = {}
            for c in feature_cols:
                lname = c.lower()
                if "crop" in lname:
                    row[c] = crop
                elif "state" in lname or "district" in lname or "region" in lname:
                    row[c] = state
                elif "soil" in lname:
                    row[c] = soil
                elif "area" in lname:
                    row[c] = area
                else:
                    row[c] = np.nan
            row_df = pd.DataFrame([row]); row_df = row_df.reindex(columns=feature_cols, fill_value=np.nan)
        except Exception:
            row_df = pd.DataFrame([{"crop":crop,"state":state,"soil":soil,"area":area}]); feature_cols = row_df.columns.tolist()
    else:
        row_df = pd.DataFrame([{"crop":crop,"state":state,"soil":soil,"area":area}]); feature_cols = row_df.columns.tolist()

    Xrow = transform_features(row_df, feature_cols, feature_encoders, feature_means)

    if show_debug:
        debug_exp = st.expander("🛠️  Debug / Details", expanded=True)
        debug_exp.write("Transformed features (Xrow):")
        debug_exp.write(Xrow)
        debug_exp.write(f"Geocoded lat/lon: {(lat, lon)}")
        debug_exp.write(f"Rainfall (12m) & temperature: {(rainfall, temperature)}")
    else:
        debug_exp = None

    seed_string = f"{crop}|{state}|{soil}|{area}|{int(rainfall) if rainfall is not None else 'NA'}|{int(temperature) if temperature is not None else 'NA'}"
    jitter = deterministic_jitter(seed_string, scale=0.05)

    # fertilizer type
    fert_type = None
    try:
        if fert_type_model is not None and fert_type_label_enc is not None:
            probs = fert_type_model.predict_proba(Xrow)
            top_idx = int(np.argmax(probs, axis=1)[0]); top_prob = float(np.max(probs))
            if top_prob >= 0.7:
                sorted_probs = np.sort(probs[0])[::-1]
                if len(sorted_probs)>1 and (sorted_probs[0]-sorted_probs[1])<0.08 and fert_type_knn is not None:
                    knn_pred = fert_type_knn.predict(Xrow).astype(int); fert_type = fert_type_label_enc.inverse_transform(knn_pred)[0]
                else:
                    fert_type = fert_type_label_enc.inverse_transform([top_idx])[0]
            else:
                if fert_type_knn is not None:
                    knn_pred = fert_type_knn.predict(Xrow).astype(int); fert_type = fert_type_label_enc.inverse_transform(knn_pred)[0]
                else:
                    top_two = np.argsort(probs[0])[::-1][:2]; pick = 0 if jitter>=0 else 1
                    fert_type = fert_type_label_enc.inverse_transform([int(top_two[pick])])[0]
        else:
            if fert_type_knn is not None and fert_type_label_enc is not None:
                knn_pred = fert_type_knn.predict(Xrow).astype(int); fert_type = fert_type_label_enc.inverse_transform(knn_pred)[0]
            else:
                fert_type = most_freq_fert_label or "Balanced NPK"
    except Exception:
        fert_type = most_freq_fert_label or "Balanced NPK"

    # fertilizer amount
    if fert_amt_model is not None:
        try: base_amt = float(fert_amt_model.predict(Xrow)[0])
        except Exception: base_amt = 80.0
    else:
        base_amt = 80.0

    soil_modifier = {"loamy":1.03,"sandy":0.9,"clay":0.95,"silty":1.02,"peaty":0.98,"chalky":0.97}.get(str(soil).lower(),1.0)
    if rainfall is None:
        rain_modifier = 1.0
    else:
        if rainfall < 400: rain_modifier = 1.12
        elif rainfall < 800: rain_modifier = 1.05
        elif rainfall < 1400: rain_modifier = 1.0
        else: rain_modifier = 0.95
    temp_modifier = 1.0 if temperature is None else max(0.9, 1 - 0.01 * abs(temperature - 24))
    amt = base_amt * soil_modifier * rain_modifier * temp_modifier
    amt = amt * (1.0 + jitter); fert_amt = round(float(max(0.1, amt)), 2)

    # pesticide type
    pest_type = None
    try:
        if pest_type_model is not None and pest_type_label_enc is not None:
            probs = pest_type_model.predict_proba(Xrow)
            top_idx = int(np.argmax(probs, axis=1)[0]); top_prob = float(np.max(probs))
            if top_prob >= 0.7:
                sorted_probs = np.sort(probs[0])[::-1]
                if len(sorted_probs)>1 and (sorted_probs[0]-sorted_probs[1])<0.08 and pest_type_knn is not None:
                    knn_pred = pest_type_knn.predict(Xrow).astype(int); pest_type = pest_type_label_enc.inverse_transform(knn_pred)[0]
                else:
                    pest_type = pest_type_label_enc.inverse_transform([top_idx])[0]
            else:
                if pest_type_knn is not None:
                    knn_pred = pest_type_knn.predict(Xrow).astype(int); pest_type = pest_type_label_enc.inverse_transform(knn_pred)[0]
                else:
                    top_two = np.argsort(probs[0])[::-1][:2]; pick = 0 if jitter>=0 else 1
                    pest_type = pest_type_label_enc.inverse_transform([int(top_two[pick])])[0]
        else:
            if pest_type_knn is not None and pest_type_label_enc is not None:
                knn_pred = pest_type_knn.predict(Xrow).astype(int); pest_type = pest_type_label_enc.inverse_transform(knn_pred)[0]
            else:
                pest_type = most_freq_pest_label or "General IPM"
    except Exception:
        pest_type = most_freq_pest_label or "General IPM"

    # pesticide amount
    if pest_amt_model is not None:
        try: base_pest = float(pest_amt_model.predict(Xrow)[0])
        except Exception: base_pest = 2.0
    else:
        base_pest = 2.0

    if rainfall is None:
        pest_rain_mod = 1.0
    else:
        if rainfall < 300: pest_rain_mod = 1.12
        elif rainfall < 800: pest_rain_mod = 1.05
        elif rainfall < 1400: pest_rain_mod = 0.98
        else: pest_rain_mod = 0.92

    pest_soil_mod = {"loamy":1.0,"sandy":1.03,"clay":0.98,"silty":1.0}.get(str(soil).lower(),1.0)
    pest_amt = base_pest * pest_rain_mod * pest_soil_mod
    pest_amt = pest_amt * (1.0 + jitter*0.6); pest_amt = round(float(max(0.01,pest_amt)),2)

    # render outputs (units included once)
    weather_place.markdown(
        """
        <div class="card">
          <div class="kpi">
            <div>
              <h3>🌦  Weather</h3>
              <div class="small">💧 Rainfall: <strong>{rain}</strong></div>
              <div class="small">🌡️ Mean temp: <strong>{temp}</strong></div>
            </div>
            <div><div class="badge">Auto</div></div>
          </div>
        </div>
        """.format(rain=(f"{round(rainfall,2)} mm") if rainfall is not None else "N/A",
                   temp=(f"{round(temperature,2)} °C") if temperature is not None else "N/A"),
        unsafe_allow_html=True,
    )

    yield_place.markdown(
        """
        <div class="card">
          <h3>🌾  Yield Prediction</h3>
          <div class="small">📈 Predicted (t/ha)</div>
          <div class="value">{yph} t/ha</div>
          <div class="small">📦 Total Production: {total} tons</div>
        </div>
        """.format(yph=yph, total=total_prod),
        unsafe_allow_html=True,
    )

    recommendations_place.markdown(
        """
        <div class="card-row">
          <div class="card">
            <h3>🧪  Fertilizer</h3>
            <div class="small">⚖️ Amount</div>
            <div class="value">{fert_amt} kg/ha</div>
            <div class="small">🏷️ Type</div>
            <div class="type-big">{fert_type}</div>
          </div>

          <div class="card">
            <h3>🐞  Pesticide</h3>
            <div class="small">⚖️ Amount</div>
            <div class="value">{pest_amt} kg/ha</div>
            <div class="small">🏷️ Type</div>
            <div class="type-big">{pest_type}</div>
          </div>
        </div>
        """.format(fert_amt=fert_amt, fert_type=fert_type, pest_amt=pest_amt, pest_type=pest_type),
        unsafe_allow_html=True,
    )

    if debug_exp is not None:
        debug_exp.markdown("<hr />")
        debug_exp.write("Additional details shown above.")

    # SHAP (optional)
    if HAS_SHAP and (fert_amt_model is not None or pest_amt_model is not None):
        try:
            st.markdown("### 🔍 Feature importance (SHAP)")
            model_for_shap = fert_amt_model if fert_amt_model is not None else pest_amt_model
            Xshap = Xrow.copy()
            explainer = shap.TreeExplainer(model_for_shap)
            shap_values = explainer.shap_values(Xshap)
            shap.initjs()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,2))
            shap.summary_plot(shap_values, Xshap, show=False)
            plt.tight_layout()
            buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight", dpi=120); buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            st.markdown(f"<img src='data:image/png;base64,{img_b64}' style='max-width:100%;'>", unsafe_allow_html=True)
        except Exception:
            st.info("SHAP explanation not available for this model/input.")

    st.markdown("---")
    st.caption("📌Developed to help farmers")

# end of file
