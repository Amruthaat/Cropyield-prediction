import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === Step 1: Load dataset ===
data = pd.read_excel("updated_crop_yield_dataset_cleaned.xlsx")

print("=== Original Data ===")
print(data.head(), "\n")

# === Step 2: Identify and encode categorical columns ===
categorical_cols = data.select_dtypes(include=['object']).columns
print("Categorical columns found:", list(categorical_cols), "\n")

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

# === Step 3: Detect target column automatically ===
possible_targets = [c for c in data.columns if 'yield' in c.lower()]
if len(possible_targets) == 0:
    raise ValueError("No column containing 'yield' found. Please specify the correct target column name.")
target_col = possible_targets[0]
print(f"Detected target column: {target_col}\n")

# === Step 4: Split features and target ===
X = data.drop(columns=[target_col])
y = data[target_col]

# === Step 5: Normalize features and target ===
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

normalized_df = pd.DataFrame(np.hstack((X_scaled, y_scaled)), columns=list(X.columns) + [target_col])
print("=== Normalized Data (first 5 rows) ===")
print(normalized_df.head(), "\n")

# === Step 6: Safe save for normalized data ===
norm_filename = "normalized_output.xlsx"
if os.path.exists(norm_filename):
    norm_filename = "normalized_output_new.xlsx"
normalized_df.to_excel(norm_filename, index=False)
print(f"✅ Normalized data saved as '{norm_filename}'\n")

# === Step 7: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# === Step 8: Train XGBoost ===
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train.ravel())

# === Step 9: Predict and denormalize ===
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_actual = scaler_y.inverse_transform(y_test)

# === Step 10: Display and safe save for results ===
results_df = pd.DataFrame({
    "Actual Yield": y_actual.flatten(),
    "Predicted Yield": y_pred.flatten()
})

print("=== Denormalized Actual vs Predicted Yields ===")
print(results_df.head(10), "\n")

results_filename = "predicted_vs_actual.xlsx"
if os.path.exists(results_filename):
    results_filename = "predicted_vs_actual_new.xlsx"
results_df.to_excel(results_filename, index=False)
print(f"✅ Denormalized predicted vs actual yields saved as '{results_filename}'\n")

# === Step 11: Evaluation metrics ===
r2 = r2_score(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100  # MAPE formula

print("=== Model Evaluation Metrics ===")
print(f"R² Score: {r2:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
