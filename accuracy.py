"""
accuracy.py
Full script to train regressors and compute accuracy metrics (R2, MAE, RMSE).
Edit INPUT_FILE and TARGET_OVERRIDE at the top as needed.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------- USER SETTINGS ----------
INPUT_FILE = r"C:\Users\ADMIN\Desktop\crop3\updated_crop_yield_dataset_cleaned.xlsx"  # <- set your path
TARGET_OVERRIDE = None     # e.g. "Yield" if auto-detect chooses wrong
TEST_SIZE = 0.20
RANDOM_STATE = 42
DO_SCALE_FOR_MLP = True
MODEL_CHOICE = "all"       # "all" or choose one: "RandomForest", "GradientBoosting", "XGBoost", "LightGBM", "CatBoost", "MLPRegressor"
SAVE_BEST_MODEL = True
# -----------------------------------

def file_not_found_handler(path):
    print(f"\nERROR: File not found: {path}")
    folder = os.path.dirname(path) if os.path.dirname(path) else os.getcwd()
    print(f"Looking in folder: {folder}\n")
    try:
        files = os.listdir(folder)
        excel_csv = [f for f in files if f.lower().endswith(('.xlsx', '.xls', '.csv'))]
        if excel_csv:
            print("Found these Excel/CSV files in that folder:")
            for f in excel_csv:
                print("  ", f)
            print("\nIf your dataset is listed above, set INPUT_FILE to the full path of that file and re-run.")
        else:
            print("No Excel/CSV files found in that folder. Place your dataset in the folder or update INPUT_FILE.")
    except Exception:
        print("Could not list files in folder (permission or path issue).")
    sys.exit(1)

# 1) Load dataset
if not os.path.exists(INPUT_FILE):
    file_not_found_handler(INPUT_FILE)

print("Loading dataset from:", INPUT_FILE)
if INPUT_FILE.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(INPUT_FILE)
else:
    df = pd.read_csv(INPUT_FILE)

print("Loaded dataframe shape:", df.shape)

# 2) Detect target column
def detect_target(df, override=None):
    if override:
        if override in df.columns:
            return override
        else:
            raise KeyError(f"TARGET_OVERRIDE '{override}' not found in columns.")
    candidates = ["yield", "production", "crop_yield", "yield_tons", "yield (tons)", "yield_kg"]
    cols = list(df.columns)
    for cand in candidates:
        for c in cols:
            if cand in c.lower():
                return c
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found to use as target. Set TARGET_OVERRIDE.")
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    return numeric_cols[-1]

try:
    target_col = detect_target(df, TARGET_OVERRIDE)
except Exception as e:
    print("Error detecting target column:", e)
    print("Available columns:", df.columns.tolist())
    sys.exit(1)

print("Using target column:", target_col)

# 3) Prepare features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Simple encoding for object/category columns (safe for tree models)
obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if obj_cols:
    print("Encoding categorical columns:", obj_cols)
    for c in obj_cols:
        X[c] = X[c].astype("category").cat.codes

# Drop rows where target is missing, simple imputation for features
mask = y.notna()
X = X.loc[mask].copy()
y = y.loc[mask].copy()

for c in X.select_dtypes(include=[np.number]).columns:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].mean())
for c in X.select_dtypes(exclude=[np.number]).columns:
    if X[c].isna().any():
        X[c] = X[c].fillna("missing")

print("Final features shape:", X.shape, "Target shape:", y.shape)

# 4) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 5) Prepare models dictionary
models = {}
if MODEL_CHOICE in ("all", "RandomForest"):
    models["RandomForest"] = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
if MODEL_CHOICE in ("all", "GradientBoosting"):
    models["GradientBoosting"] = GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE)
if MODEL_CHOICE in ("all", "MLPRegressor"):
    models["MLPRegressor"] = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=1000, random_state=RANDOM_STATE)

# optional libs
if MODEL_CHOICE in ("all", "XGBoost"):
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
    except Exception:
        print("xgboost not installed - skipping XGBoost (pip install xgboost)")

if MODEL_CHOICE in ("all", "LightGBM"):
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    except Exception:
        print("lightgbm not installed - skipping LightGBM (pip install lightgbm)")

if MODEL_CHOICE in ("all", "CatBoost"):
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(iterations=1000, verbose=0, random_state=RANDOM_STATE)
    except Exception:
        print("catboost not installed - skipping CatBoost (pip install catboost)")

if not models:
    print("No models selected / available. Exiting.")
    sys.exit(1)

print("Models to run:", list(models.keys()))

# 6) Train, predict, compute metrics
results = []
pred_store = {}
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

for name, mdl in models.items():
    print(f"\n--- Training {name} ---")
    try:
        if name == "MLPRegressor" and DO_SCALE_FOR_MLP:
            pipeline = Pipeline([
                ("scaler", ColumnTransformer([("num", StandardScaler(), numeric_cols)], remainder="passthrough")),
                ("model", mdl)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            pred_store[name] = (pipeline, y_pred)
        else:
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)
            pred_store[name] = (mdl, y_pred)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        acc_percent = r2 * 100

        print(f"{name}: R2={r2:.4f}, Accuracy%={acc_percent:.2f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        results.append({"model": name, "r2": r2, "accuracy_percent": acc_percent, "mae": mae, "rmse": rmse})
    except Exception as e:
        print(f"Failed {name}: {e}")

if not results:
    print("No successful model runs. Exiting.")
    sys.exit(1)

# 7) Save metrics and choose best
results_df = pd.DataFrame(results).sort_values(by=["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)
results_df.index += 1
results_df.to_csv("accuracy_results.csv", index=False)
print("\nSaved accuracy_results.csv")
print(results_df)

best = results_df.iloc[0]
best_name = best["model"]
print(f"\nBest model: {best_name} (R2={best['r2']:.4f}, RMSE={best['rmse']:.4f})")

# save predictions for best
best_obj, best_preds = pred_store[best_name]
test_df = X_test.copy()
test_df[target_col + "_true"] = y_test
test_df[target_col + "_pred"] = best_preds
test_df.to_csv("best_model_predictions.csv", index=False)
test_df.to_excel("best_model_predictions.xlsx", index=False)
print("Saved best_model_predictions.csv and .xlsx")

# 8) Optionally save the best model
if SAVE_BEST_MODEL:
    try:
        import joblib
        joblib.dump(best_obj, "best_model.joblib")
        print("Saved best model to best_model.joblib")
    except Exception:
        print("joblib not available - to save model install joblib (pip install joblib)")

print("\nDone. Check accuracy_results.csv and best_model_predictions.csv")
