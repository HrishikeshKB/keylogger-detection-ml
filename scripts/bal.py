import pandas as pd
import os

raw_folder = "data/raw"
files = os.listdir(raw_folder)

all_data = []

for file in files:
    if file.endswith(".csv"):
        path = os.path.join(raw_folder, file)
        df = pd.read_csv(path)
        all_data.append(df)

if not all_data:
    print("No CSV files found in data/raw/")
else:
    final_df = pd.concat(all_data, ignore_index=True)

    total = len(final_df)
    normal = len(final_df[final_df["label"] == 0])
    suspicious = len(final_df[final_df["label"] == 1])

    print("----- DATASET STATUS -----")
    print(f"Total Samples: {total}")
    print(f"Normal (0): {normal}")
    print(f"Suspicious (1): {suspicious}")

    if normal == 0:
        print("\n⚠ No normal data yet.")
    if suspicious == 0:
        print("\n⚠ No suspicious data yet.")

    if normal > 0 and suspicious > 0:
        ratio = suspicious / normal
        print(f"\nSuspicious to Normal Ratio: {ratio:.2f}")