import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_path = os.path.join(BASE_DIR, "data", "raw")
output_file = os.path.join(raw_path, "normal_synthetic.csv")

np.random.seed(42)

rows = []

for i in range(100):

    keystroke_rate = np.clip(np.random.normal(1.2, 0.8), 0, 4)
    avg_key_gap = np.clip(np.random.normal(0.20, 0.12), 0, 1)
    cpu_usage = np.clip(np.random.normal(16, 6), 5, 35)
    memory_usage = np.clip(np.random.normal(60, 6), 45, 80)
    process_count = np.clip(np.random.normal(270, 25), 200, 330)
    window_switches = np.clip(np.random.normal(7, 4), 0, 20)

    rows.append([
        keystroke_rate,
        avg_key_gap,
        cpu_usage,
        memory_usage,
        process_count,
        window_switches,
        0
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

print("Human-like normal synthetic data generated.")