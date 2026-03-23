import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import sys
import joblib
# Load dataset

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")

df = pd.read_csv(data_path)

print("Loaded dataset shape:", df.shape)

print("\n=== DATA OVERVIEW ===")
print(df.describe())

print("\n=== CLASS-WISE AVERAGES ===")
print(df.groupby("label").mean())

# Features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RandomForest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("\nRandomForest model trained.")

# Train Isolation Forest
iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

iso_model.fit(X_train)

print("Isolation Forest model trained.")

# Save models
rf_model_path = os.path.join(BASE_DIR, "models", "keylogger_model.pkl")
iso_model_path = os.path.join(BASE_DIR, "models", "anomaly_model.pkl")

joblib.dump(rf_model, rf_model_path)
joblib.dump(iso_model, iso_model_path)

print("RandomForest model saved at:", rf_model_path)
print("IsolationForest model saved at:", iso_model_path)

# Predictions (RandomForest)
y_pred = rf_model.predict(X_test)

print("\n=== MODEL PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
print("\n=== FEATURE IMPORTANCE ===")

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

print(feature_importance.sort_values(ascending=False))

# SHAP EXPLAINABILITY
print("\n=== RUNNING SHAP ANALYSIS ===")

explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)

# Handle binary classification output
if isinstance(shap_values.values, np.ndarray) and len(shap_values.values.shape) == 3:
    shap_class1 = shap.Explanation(
        values=shap_values.values[:, :, 1],
        base_values=shap_values.base_values[:, 1],
        data=shap_values.data,
        feature_names=X_test.columns
    )
else:
    shap_class1 = shap_values

# Global explanation
print("Displaying SHAP Summary Plot...")
shap.plots.beeswarm(shap_class1)

# Explain single prediction
print("\nExplaining one sample prediction...")

sample_index = 0

print("Sample values:\n", X_test.iloc[sample_index])
print("Actual label:", y_test.iloc[sample_index])
print("Predicted label:", rf_model.predict(X_test.iloc[[sample_index]])[0])

shap.plots.waterfall(shap_class1[sample_index])

plt.show()