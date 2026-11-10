# compare_linear_rf_xgb_mape.py
"""
Compare multiple regressors: LinearRegression, RandomForest, GradientBoosting,
XGBoost (if available), and LightGBM (if available).
Select best model by R2 (tie-breaker RMSE). Also computes MAPE.
Robust imputing, scaling for linear model, and safe saving of results.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # ADDED GB
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ---------- USER SETTINGS ----------
INPUT_FILE = r"C:\Users\ADMIN\Desktop\crop3\updated_crop_yield_dataset_cleaned.xlsx"
TARGET_OVERRIDE = None
TEST_SIZE = 0.2
RANDOM_STATE = 42
# -----------------------------------

def MAPE(y_true, y_pred):
    """Handles divide-by-zero safely."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def file_not_found_handler(path):
    print(f"\nERROR: File not found: {path}")
    folder = os.path.dirname(path) if os.path.dirname(path) else os.getcwd()
    print(f"Looking in folder: {folder}\n")
    try:
        files = os.listdir(folder)
        excel_csv = [f for f in files if f.lower().endswith(('.xlsx', '.xls', '.csv'))]
        if excel_csv:
            print("Found Excel/CSV files:")
            for f in excel_csv:
                print("   ", f)
        else:
            print("No Excel/CSV files found in that folder.")
    except Exception:
        print("Could not list files in folder.")
    sys.exit(1)

# 1) Load dataset
if not os.path.exists(INPUT_FILE):
    file_not_found_handler(INPUT_FILE)

print("Loading dataset from:", INPUT_FILE)
df = pd.read_excel(INPUT_FILE) if INPUT_FILE.lower().endswith((".xlsx", ".xls")) else pd.read_csv(INPUT_FILE)
print("Loaded:", df.shape)

# 2) Detect target column
def detect_target(df, override=None):
    if override:
        if override in df.columns:
            return override
        raise KeyError(f"TARGET_OVERRIDE '{override}' not found.")
    candidates = ["yield", "production", "crop_yield", "yield_tons", "yield (tons)", "yield_kg"]
    for cand in candidates:
        for c in df.columns:
            if cand in c.lower():
                return c
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols[-1]

target_col = detect_target(df, TARGET_OVERRIDE)
print("Using target column:", target_col)

# 3) Prepare X and y
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Convert target to numeric if needed
if not pd.api.types.is_numeric_dtype(y):
    y = pd.to_numeric(y, errors="coerce")

mask = y.notna()
X = X.loc[mask].copy()
y = y.loc[mask].copy()

# Encode categoricals
obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for c in obj_cols:
    X[c] = X[c].astype("category").cat.codes.replace({-1: np.nan})

# Impute missing
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

if numeric_cols:
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
if non_numeric_cols:
    X[non_numeric_cols] = cat_imputer.fit_transform(X[non_numeric_cols])

print("Final X:", X.shape, "| y:", y.shape)

# 4) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# 5) Models
models = {}

# Linear Regression with scaling
if numeric_cols:
    preproc_lr = ColumnTransformer(
        [("num", Pipeline([("scaler", StandardScaler())]), numeric_cols)],
        remainder="passthrough"
    )
    lr_model = Pipeline([("preproc", preproc_lr), ("model", LinearRegression())])
else:
    lr_model = LinearRegression()

models["LinearRegression"] = lr_model

# Random Forest
models["RandomForest"] = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE
)

# Gradient Boosting (sklearn)
models["GradientBoosting"] = GradientBoostingRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE
)

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    models["XGBoost"] = XGBRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
except Exception:
    print("xgboost not installed -> skipping XGBoost")

# Optional LightGBM
try:
    from lightgbm import LGBMRegressor
    models["LightGBM"] = LGBMRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
except Exception:
    print("lightgbm not installed -> skipping LightGBM")

# 6) Train & evaluate
results = []
predictions_store = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape_val = MAPE(y_test, y_pred)

        print(f"{name} -> R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape_val:.2f}%")

        results.append({
            "model": name,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "mape": mape_val
        })
        predictions_store[name] = (model, y_pred)

    except Exception as e:
        print(f"{name} failed:", e)

if not results:
    print("No models trained successfully.")
    sys.exit(1)

# 7) Save comparison
results_df = pd.DataFrame(results).sort_values(
    by=["r2", "rmse"],
    ascending=[False, True]
).reset_index(drop=True)

print("\nModel comparison:")
print(results_df)
results_df.to_csv("model_comparison.csv", index=False)

# 8) Choose best
best_row = results_df.iloc[0]
best_name = best_row["model"]
best_model, best_preds = predictions_store[best_name]

print(f"\nBest model: {best_name}   R2={best_row['r2']:.4f}   RMSE={best_row['rmse']:.4f}")

# 9) Save predictions
test_out = X_test.reset_index(drop=True).copy()
test_out[target_col + "_true"] = y_test.reset_index(drop=True)
test_out[target_col + "_pred"] = best_preds

# safe MAPE per row (avoid divide-by-zero)
true_vals = test_out[target_col + "_true"].values
pred_vals = test_out[target_col + "_pred"].values
row_mape = np.full_like(true_vals, fill_value=np.nan, dtype=float)
nonzero_mask = true_vals != 0
row_mape[nonzero_mask] = np.abs(true_vals[nonzero_mask] - pred_vals[nonzero_mask]) / np.abs(true_vals[nonzero_mask]) * 100
test_out["MAPE"] = row_mape

test_out.to_csv("best_model_predictions.csv", index=False)
try:
    test_out.to_excel("best_model_predictions.xlsx", index=False)
except Exception:
    print("Warning: Could not save Excel file, only CSV saved.")

# 10) Save best model
try:
    import joblib
    joblib.dump(best_model, "best_model.joblib")
    print("Saved best model to best_model.joblib")
except Exception:
    print("Install joblib to enable model saving.")

print("\nDone.")
