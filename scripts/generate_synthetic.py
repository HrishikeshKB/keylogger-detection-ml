import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_path = os.path.join(BASE_DIR, "data", "raw")
output_file = os.path.join(raw_path, "suspicious_synthetic.csv")

np.random.seed(42)

rows = []

for i in range(250):  # adjust count if needed
    
    keystroke_rate = np.clip(np.random.normal(2.0, 1.0), 0, 6)
    avg_key_gap = np.clip(np.random.normal(0.15, 0.1), 0, 1)
    cpu_usage = np.clip(np.random.normal(22, 6), 10, 40)
    memory_usage = np.clip(np.random.normal(70, 5), 55, 85)
    process_count = np.clip(np.random.normal(310, 25), 250, 380)
    window_switches = np.clip(np.random.normal(10, 4), 0, 25)

    rows.append([
        keystroke_rate,
        avg_key_gap,
        cpu_usage,
        memory_usage,
        process_count,
        window_switches,
        1
    ])

columns = [
    "keystroke_rate",
    "avg_key_gap",
    "cpu_usage",
    "memory_usage",
    "process_count",
    "window_switches",
    "label"
]

df = pd.DataFrame(rows, columns=columns)
df.to_csv(output_file, index=False)

print("New human-like suspicious data generated.")