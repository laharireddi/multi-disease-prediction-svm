"""
Multi-Disease Prediction System - Streamlit App
Run with: streamlit run app.py
Expects trained models/scalers in ../models/ (produced by the train_*.py scripts)
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_pickle(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- Sidebar Navigation ----------
st.sidebar.title("🧬 Multi-Disease Prediction System")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🩺 Diabetes Prediction", "❤️ Heart Disease Prediction",
     "🧠 Parkinson's Prediction", "📊 Dataset Information",
     "📈 Model Performance", "ℹ️ About"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Built by Lahari Reddi")
st.sidebar.caption("B.Tech CSE | Andhra University")
# =========================================================
# HOME
# =========================================================
if page == "🏠 Home":
    st.title("🩺 Multi-Disease Prediction System")
    st.markdown("##### *SVM-Based Health Diagnostics System*")
    st.markdown("---")
    st.markdown("""
    Welcome! This application uses **Support Vector Machine (SVM)** models
    to predict the likelihood of three conditions:

    - 🩺 **Diabetes**
    - ❤️ **Heart Disease**
    - 🧠 **Parkinson's Disease**

    Use the sidebar to navigate between prediction pages, explore the
    datasets used, and review model performance metrics.
    """)
    st.info("👈 Select a disease from the sidebar to get started.")
    st.warning("""
    ⚠️ **Disclaimer:** This tool is built for educational and academic
    purposes as part of an AI/ML internship project. It is **not** a
    substitute for professional medical diagnosis or advice. Please
    consult a qualified healthcare provider for any medical concerns.
    """)
# =========================================================
# DIABETES PREDICTION
# =========================================================
elif page == "🩺 Diabetes Prediction":
    st.title("🩺 Diabetes Prediction")
    model = load_pickle("diabetes_model.pkl")
    scaler = load_pickle("diabetes_scaler.pkl")

    if model is None:
        st.warning("Model not found. Run `train_diabetes.py` first to generate it.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0, 300, 120)
            bp = st.number_input("Blood Pressure", 0, 200, 70)
        with col2:
            skin = st.number_input("Skin Thickness", 0, 100, 20)
            insulin = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        with col3:
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 1, 120, 30)

        if st.button("Predict Diabetes"):
            features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0][1]

            if prediction == 1:
                st.error(f"⚠️ High likelihood of Diabetes ({proba*100:.1f}% confidence)")
            else:
                st.success(f"✅ Low likelihood of Diabetes ({(1-proba)*100:.1f}% confidence)")

# =========================================================
# HEART DISEASE PREDICTION
# =========================================================
elif page == "❤️ Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction")
    model = load_pickle("heart_model.pkl")
    scaler = load_pickle("heart_scaler.pkl")
    metrics = load_pickle("heart_metrics.pkl")

    if model is None:
        st.warning("Model not found. Run `train_heart.py` first to generate it.")
    else:
        st.caption("Enter values matching your dataset's feature order.")
        feature_order = metrics["feature_order"] if metrics else [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
        inputs = []
        cols = st.columns(3)
        for i, feat in enumerate(feature_order):
            with cols[i % 3]:
                val = st.number_input(feat, value=0.0, key=f"heart_{feat}")
                inputs.append(val)

        if st.button("Predict Heart Disease"):
            features = np.array([inputs])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0][1]

            if prediction == 1:
                st.error(f"⚠️ High likelihood of Heart Disease ({proba*100:.1f}% confidence)")
            else:
                st.success(f"✅ Low likelihood of Heart Disease ({(1-proba)*100:.1f}% confidence)")

# =========================================================
# PARKINSON'S PREDICTION
# =========================================================
elif page == "🧠 Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction")
    model = load_pickle("parkinsons_model.pkl")
    scaler = load_pickle("parkinsons_scaler.pkl")
    metrics = load_pickle("parkinsons_metrics.pkl")

    if model is None:
        st.warning("Model not found. Run `train_parkinsons.py` first to generate it.")
    else:
        st.caption("Enter voice measurement values matching your dataset's feature order.")
        feature_order = metrics["feature_order"] if metrics else []
        inputs = []
        cols = st.columns(3)
        for i, feat in enumerate(feature_order):
            with cols[i % 3]:
                val = st.number_input(feat, value=0.0, key=f"pd_{feat}")
                inputs.append(val)

        if st.button("Predict Parkinson's"):
            features = np.array([inputs])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0][1]

            if prediction == 1:
                st.error(f"⚠️ High likelihood of Parkinson's ({proba*100:.1f}% confidence)")
            else:
                st.success(f"✅ Low likelihood of Parkinson's ({(1-proba)*100:.1f}% confidence)")

# =========================================================
# DATASET INFORMATION
# =========================================================
elif page == "📊 Dataset Information":
    st.title("📊 Dataset Information")

    st.subheader("🩺 Diabetes Dataset (Pima Indians)")
    st.markdown("""
    - **Source:** UCI / Kaggle - Pima Indians Diabetes Database
    - **Samples:** 768 | **Features:** 8
    - **Target:** Outcome (1 = Diabetic, 0 = Non-diabetic)
    - **Key features:** Glucose, BMI, Age, Insulin, Blood Pressure
    """)

    st.subheader("❤️ Heart Disease Dataset (UCI / Cleveland)")
    st.markdown("""
    - **Source:** UCI Machine Learning Repository
    - **Samples:** ~303 | **Features:** 13
    - **Target:** target (1 = Presence of disease, 0 = Absence)
    - **Key features:** Chest pain type, resting BP, cholesterol, max heart rate
    """)

    st.subheader("🧠 Parkinson's Dataset (UCI)")
    st.markdown("""
    - **Source:** UCI Machine Learning Repository
    - **Samples:** 195 | **Features:** 22 voice measurements
    - **Target:** status (1 = Parkinson's, 0 = Healthy)
    - **Key features:** Jitter, Shimmer, HNR, and other vocal frequency measures
    """)

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("Accuracy achieved by each SVM model on held-out test data.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    diseases = [("🩺 Diabetes", "diabetes", col1), ("❤️ Heart Disease", "heart", col2),
                ("🧠 Parkinson's", "parkinsons", col3)]

    for label, key, col in diseases:
        metrics = load_pickle(f"{key}_metrics.pkl")
        with col:
            st.subheader(label)
            if metrics:
                st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
                with st.expander("Best Parameters"):
                    st.json(metrics["best_params"])
            else:
                st.warning(f"Train the {key} model first.")

    st.markdown("---")
    st.markdown("""
    **Note:** All models use an SVM classifier with an RBF kernel,
    selected through hyperparameter tuning for optimal performance
    on each disease's respective dataset.
    """)
# =========================================================
# ABOUT
# =========================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    **Multi-Disease Prediction System** is an SVM-based machine learning
    application built as part of the Robokwik Prabandhan AI/ML/DS internship
    program (IIT Kanpur).

    It applies Support Vector Machines with kernel methods to predict
    Diabetes, Heart Disease, and Parkinson's Disease from clinical and
    voice-based measurements.

    **Built by:** Lahari Reddi
    **Tech stack:** Python, scikit-learn, Streamlit, pandas, seaborn
    """)
