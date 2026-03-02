import time
import psutil
import pandas as pd
import keyboard
import win32gui

data = []

keystroke_count = 0
key_times = []
window_switches = 0
last_window = None
start_time = time.time()

print("Logging started... Press ESC to stop.")

def on_key_event(event):
    global keystroke_count, key_times
    keystroke_count += 1
    key_times.append(time.time())

keyboard.on_press(on_key_event)

label = 0   # 0 = Normal , 1 = Suspicious

try:
    while True:
        time.sleep(5)

        current_time = time.time()
        interval = current_time - start_time

        # Keystroke rate
        keystroke_rate = keystroke_count / interval if interval > 0 else 0

        # Average key gap
        if len(key_times) > 1:
            gaps = [key_times[i] - key_times[i-1] for i in range(1, len(key_times))]
            avg_gap = sum(gaps) / len(gaps)
        else:
            avg_gap = 0

        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        process_count = len(psutil.pids())

        # Window switching
        current_window = win32gui.GetForegroundWindow()
        if last_window is not None and current_window != last_window:
            window_switches += 1
        last_window = current_window

        data.append([
            keystroke_rate,
            avg_gap,
            cpu_usage,
            memory_usage,
            process_count,
            window_switches,
            label
        ])

        print(f"Rate:{keystroke_rate:.2f}, Gap:{avg_gap:.3f}, CPU:{cpu_usage}%, Mem:{memory_usage}%, Proc:{process_count}, Switch:{window_switches}")

        keystroke_count = 0
        key_times = []
        start_time = current_time

        if keyboard.is_pressed("esc"):
            break

except KeyboardInterrupt:
    pass

df = pd.DataFrame(data, columns=[
    "keystroke_rate",
    "avg_key_gap",
    "cpu_usage",
    "memory_usage",
    "process_count",
    "window_switches",
    "label"
])

df.to_csv("data/raw/behavior_data.csv", index=False)

print("Data saved to data/raw/behavior_data.csv")