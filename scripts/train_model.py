import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# -----------------------------
# Load dataset
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")

df = pd.read_csv(data_path)

print("Loaded dataset shape:", df.shape)

print("\n=== DATA OVERVIEW ===")
print(df.describe())

print("\n=== CLASS-WISE AVERAGES ===")
print(df.groupby("label").mean())

# -----------------------------
# Features and labels
# -----------------------------
X = df.drop("label", axis=1)
y = df["label"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

print("\n=== MODEL PERFORMANCE ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Feature Importance
# -----------------------------
print("\n=== FEATURE IMPORTANCE ===")
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)
print(feature_importance.sort_values(ascending=False))

# -----------------------------
# SHAP EXPLAINABILITY (FINAL FIX)
# -----------------------------
print("\n=== RUNNING SHAP ANALYSIS ===")

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# For binary classification, select class 1 (Suspicious)
if isinstance(shap_values.values, np.ndarray) and len(shap_values.values.shape) == 3:
    shap_class1 = shap.Explanation(
        values=shap_values.values[:, :, 1],
        base_values=shap_values.base_values[:, 1],
        data=shap_values.data,
        feature_names=X_test.columns
    )
else:
    shap_class1 = shap_values

# Global beeswarm plot (Suspicious class)
print("Displaying SHAP Summary Plot...")
shap.plots.beeswarm(shap_class1)

# Single prediction explanation
print("\nExplaining one sample prediction...")

sample_index = 0

print("Sample values:\n", X_test.iloc[sample_index])
print("Actual label:", y_test.iloc[sample_index])
print("Predicted label:", model.predict(X_test.iloc[[sample_index]])[0])

shap.plots.waterfall(shap_class1[sample_index])

plt.show()