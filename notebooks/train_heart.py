"""
Heart Disease Prediction - SVM Model Training
Expected dataset: data/heart.csv (UCI Heart Disease / Cleveland dataset)
Typical columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                  exang, oldpeak, slope, ca, thal, target
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

DATA_PATH = "../data/heart.csv"
MODEL_DIR = "../models"
REPORT_DIR = "../reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------- 1. Load Data ----------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())

# ---------- 2. Preprocessing ----------
# Drop duplicate rows if any, check for nulls
df = df.drop_duplicates()
df = df.dropna()

target_col = "target" if "target" in df.columns else df.columns[-1]

# ---------- 3. Feature Selection / Correlation ----------
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heart Disease Dataset - Feature Correlation")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/heart_correlation.png")
plt.close()

# ---------- 4. Visualization ----------
plt.figure(figsize=(6, 4))
sns.countplot(x=target_col, data=df)
plt.title("Heart Disease - Class Distribution")
plt.savefig(f"{REPORT_DIR}/heart_class_distribution.png")
plt.close()

# ---------- 5. Train/Test Split ----------
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 6. Scaling ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 7. SVM with GridSearch ----------
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.001],
    "kernel": ["rbf", "linear"]
}
grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)

# ---------- 8. Evaluation ----------
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Heart Disease - Confusion Matrix")
plt.savefig(f"{REPORT_DIR}/heart_confusion_matrix.png")
plt.close()

# ---------- 9. Save Model + Scaler ----------
with open(f"{MODEL_DIR}/heart_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(f"{MODEL_DIR}/heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

metrics = {"accuracy": acc, "best_params": grid.best_params_, "feature_order": list(X.columns)}
with open(f"{MODEL_DIR}/heart_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("Heart model, scaler, and metrics saved to /models")
