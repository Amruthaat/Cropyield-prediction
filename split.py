import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------
# Load your dataset
# ------------------------------
file_path = "updated_crop_yield_dataset_cleaned.xlsx"   # change your file name here
df = pd.read_excel(file_path)

# ------------------------------
# Train-Test Split
# ------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ------------------------------
# Save as CSV
# ------------------------------
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

# ------------------------------
# Save as Excel XLSX
# ------------------------------
train_df.to_excel("train_data.xlsx", index=False)
test_df.to_excel("test_data.xlsx", index=False)

print("Train and Test datasets created & saved successfully!")
