# normalize_encode_to_excel.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import inspect

# ---------- CONFIG ----------
INPUT_FILE = "updated_crop_yield_dataset_cleaned.xlsx"
OUTPUT_XLSX = "processed_dataset.xlsx"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
# ----------------------------

def main():
    # 1. Load dataset
    df = pd.read_excel(INPUT_FILE)
    print(f"✅ Loaded {INPUT_FILE}: {df.shape[0]} rows × {df.shape[1]} cols")

    # 2. Detect numeric & categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("\n📊 Numeric columns:", numeric_cols)
    print("📁 Categorical columns:", categorical_cols)

    # 3. Build OneHotEncoder compatible with all sklearn versions
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    else:
        onehot = OneHotEncoder(sparse=False, handle_unknown="ignore")

    # 4. Pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", onehot)
    ])

    # 5. ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # 6. Fit & transform
    print("\n🔄 Fitting and transforming dataset...")
    X_processed = preprocessor.fit_transform(df)

    # 7. Generate output column names
    out_feature_names = []
    out_feature_names.extend(numeric_cols)

    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
        out_feature_names.extend(cat_feature_names)

    # 8. Create processed DataFrame
    df_processed = pd.DataFrame(X_processed, columns=out_feature_names, index=df.index)

    # 9. Save to Excel file
    df_processed.to_excel(OUTPUT_XLSX, index=False)
    print(f"\n💾 Processed dataset saved to: {OUTPUT_XLSX}")

    # 10. Save preprocessing artifacts
    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.joblib")
    joblib.dump(numeric_cols, ARTIFACT_DIR / "numeric_cols.joblib")
    joblib.dump(categorical_cols, ARTIFACT_DIR / "categorical_cols.joblib")
    joblib.dump(out_feature_names, ARTIFACT_DIR / "processed_feature_names.joblib")

    print(f"🧠 Artifacts saved in: {ARTIFACT_DIR.resolve()}")
    print(f"✅ Final processed shape: {df_processed.shape[0]} rows × {df_processed.shape[1]} cols")

if __name__ == "__main__":
    main()
