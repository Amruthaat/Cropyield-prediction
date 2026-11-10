# load_full_dataset.py
import pandas as pd

# 1. File path
file_path = "updated_crop_yield_dataset_cleaned.xlsx"

# 2. Load dataset
df = pd.read_excel(file_path)

# 3. Display confirmation
print("✅ Dataset loaded successfully!")
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
print("\n--- Column Names ---")
print(df.columns.tolist())

# 4. Set display options to show all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# 5. Print the full dataset
print("\n--- Full Dataset ---")
print(df)

# 6. (Optional) Save a CSV copy if needed
# df.to_csv("full_dataset_output.csv", index=False)
