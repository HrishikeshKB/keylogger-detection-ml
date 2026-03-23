import psutil
import time
import joblib
import pandas as pd
import os
from pynput import keyboard
import datetime
import pygetwindow as gw
import matplotlib.pyplot as plt
from collections import deque
from tkinter import messagebox, Tk

# Locate models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rf_model_path = os.path.join(BASE_DIR, "models", "keylogger_model.pkl")
iso_model_path = os.path.join(BASE_DIR, "models", "anomaly_model.pkl")

log_path = os.path.join(BASE_DIR, "alerts.log")

rf_model = joblib.load(rf_model_path)
iso_model = joblib.load(iso_model_path)

# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Keyboard tracking
key_count = 0
last_key_time = None
gaps = []

def on_press(key):
    global key_count, last_key_time, gaps

    now = time.time()
    key_count += 1

    if last_key_time is not None:
        gaps.append(now - last_key_time)

    last_key_time = now

listener = keyboard.Listener(on_press=on_press)
listener.start()

interval = 5

# -----------------------------
# GRAPH STORAGE
# -----------------------------
score_history = deque(maxlen=100)

# -----------------------------
# ALERT CONTROL
# -----------------------------
alert_triggered = False

def show_alert(score):
    root = Tk()
    root.withdraw()

    messagebox.showwarning(
        "⚠ SECURITY ALERT",
        f"High Threat Detected!\n\nScore: {score:.1f}%\n\nPossible Keylogger Activity"
    )

    root.destroy()

print(CYAN + "\nKeylogger Detection Monitor Started" + RESET)

# -----------------------------
# MAIN LOOP
# -----------------------------
try:
    while True:

        start_time = time.time()

        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        process_count = len(psutil.pids())

        # Window switch detection
        window_switches = 0
        previous_window = None

        for _ in range(interval * 2):

            try:
                window = gw.getActiveWindow()
                current_window = window.title if window else None
            except:
                current_window = None

            if previous_window and current_window != previous_window:
                window_switches += 1

            previous_window = current_window
            time.sleep(0.5)

        # Keyboard features
        keystroke_rate = key_count / interval
        avg_key_gap = sum(gaps)/len(gaps) if gaps else 0

        burst_detected = keystroke_rate > 12

        key_count = 0
        gaps = []

        data = pd.DataFrame([[
            keystroke_rate,
            avg_key_gap,
            cpu_usage,
            memory_usage,
            process_count,
            window_switches
        ]], columns=[
            "keystroke_rate",
            "avg_key_gap",
            "cpu_usage",
            "memory_usage",
            "process_count",
            "window_switches"
        ])

        # Predictions
        rf_prediction = rf_model.predict(data)[0]
        rf_probability = rf_model.predict_proba(data)[0][1]

        iso_prediction = iso_model.predict(data)[0]
        anomaly_detected = iso_prediction == -1

        score_percent = rf_probability * 100

        # Save for graph
        score_history.append(score_percent)

        # Popup Alert Logic (NO SPAM)
        if score_percent >= 70 and not alert_triggered:
            show_alert(score_percent)
            alert_triggered = True

        if score_percent < 60:
            alert_triggered = False

        # Threat Meter
        bar_length = 30
        filled = int(rf_probability * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        if score_percent < 40:
            color = GREEN
            status = "LOW RISK"
        elif score_percent < 70:
            color = YELLOW
            status = "MODERATE"
        else:
            color = RED
            status = "HIGH RISK"

        if rf_prediction == 1 and anomaly_detected:
            status = "HIGH RISK"
            color = RED
        elif rf_prediction == 1 or anomaly_detected:
            status = "MODERATE"
            color = YELLOW

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Clear screen
        os.system("cls" if os.name == "nt" else "clear")

        print(CYAN + "===================================")
        print("   KEYLOGGER DETECTION MONITOR")
        print("===================================" + RESET)

        print(f"\nTime             : {timestamp}")

        print("\nSYSTEM METRICS")
        print(f"CPU Usage        : {cpu_usage:.1f}%")
        print(f"Memory Usage     : {memory_usage:.1f}%")
        print(f"Process Count    : {process_count}")

        print("\nKEYBOARD ACTIVITY")
        print(f"Keystroke Rate   : {keystroke_rate:.2f} keys/sec")
        print(f"Average Key Gap  : {avg_key_gap:.3f} sec")
        print(f"Window Switches  : {window_switches}")

        print("\nML ANALYSIS")

        print("Threat Meter")
        print(color + bar + RESET)
        print(f"Score: {score_percent:.1f}%")

        print("\nRisk Levels")
        print("LOW <40% | MODERATE 40-70% | HIGH >70%")

        print(f"\nRandomForest     : {'Suspicious' if rf_prediction else 'Normal'}")
        print(f"IsolationForest  : {'Anomaly' if anomaly_detected else 'Normal'}")

        print(color + f"\nFINAL STATUS     : {status}" + RESET)

        if burst_detected:
            print(RED + "\n⚡ Extreme Typing Burst Detected!" + RESET)

        if rf_prediction == 1 or anomaly_detected:
            with open(log_path, "a") as log:
                log.write(
                    f"{timestamp} | {status} | "
                    f"Score:{score_percent:.1f}% | "
                    f"CPU:{cpu_usage}% MEM:{memory_usage}% PROC:{process_count} | "
                    f"Rate:{keystroke_rate:.2f} Gap:{avg_key_gap:.3f} Switch:{window_switches}\n"
                )

        elapsed = time.time() - start_time

        if elapsed < interval:
            time.sleep(interval - elapsed)

# -----------------------------
# AFTER STOP (GRAPH)
# -----------------------------
except KeyboardInterrupt:

    print("\nMonitoring stopped. Generating graph...")

    plt.plot(score_history, marker='o')
    plt.title("Threat Score Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.grid()

    plt.show()