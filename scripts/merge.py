import pandas as pd
import glob

# Path to all CSV files inside data/raw
file_paths = glob.glob("data/raw/*.csv")

# Read and combine
df_list = [pd.read_csv(file) for file in file_paths]
merged_df = pd.concat(df_list, ignore_index=True)

# Shuffle rows (VERY important)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save merged dataset
merged_df.to_csv("data/processed/final_dataset.csv", index=False)

print("Data merged successfully!")
print("Total samples:", len(merged_df))
print(merged_df["label"].value_counts())