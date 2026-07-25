"""
Diabetes Prediction - SVM Model Training
Expected dataset: data/diabetes.csv  (Pima Indians Diabetes Dataset)
Columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
         BMI, DiabetesPedigreeFunction, Age, Outcome
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

# ---------- Paths ----------
DATA_PATH = "../data/diabetes.csv"
MODEL_DIR = "../models"
REPORT_DIR = "../reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------- 1. Load Data ----------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())

# ---------- 2. Preprocessing ----------
# Pima dataset uses 0 as a placeholder for missing values in these columns
cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_invalid_zero:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# ---------- 3. Feature Selection / Correlation ----------
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Diabetes Dataset - Feature Correlation")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/diabetes_correlation.png")
plt.close()

# ---------- 4. Visualization ----------
plt.figure(figsize=(6, 4))
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes - Class Distribution")
plt.savefig(f"{REPORT_DIR}/diabetes_class_distribution.png")
plt.close()

# ---------- 5. Train/Test Split ----------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 6. Scaling (critical for SVM) ----------
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
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
disp.plot(cmap="Blues")
plt.title("Diabetes - Confusion Matrix")
plt.savefig(f"{REPORT_DIR}/diabetes_confusion_matrix.png")
plt.close()

# ---------- 9. Save Model + Scaler ----------
with open(f"{MODEL_DIR}/diabetes_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(f"{MODEL_DIR}/diabetes_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save metrics for the Streamlit "Model Performance" page
metrics = {"accuracy": acc, "best_params": grid.best_params_}
with open(f"{MODEL_DIR}/diabetes_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("Diabetes model, scaler, and metrics saved to /models")
