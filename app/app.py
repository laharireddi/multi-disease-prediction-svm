"""
Multi-Disease Prediction System - Streamlit dashboard
Run from the project root:  python -m streamlit run app/app.py

Expects, relative to this file:
  ../models/  trained model, scaler and metrics (.pkl) from the notebooks/train_*.py scripts
  ../data/    the three CSV datasets (used for reference charts and sample patients)
"""

import os
import pickle
import warnings
from datetime import datetime
from html import escape as esc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="auto",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK = "#0E2230"
INK_2 = "#506676"
LINE = "#D9E1E4"
CANVAS = "#F2F5F5"
TEAL = "#0B7A75"
RISK = "#B23A2E"
CLEAR = "#2E7D5B"
AMBER = "#B7791F"
FONT = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {{
  --ink: {INK}; --ink-2: {INK_2}; --line: {LINE}; --canvas: {CANVAS};
  --teal: {TEAL}; --risk: {RISK}; --clear: {CLEAR}; --amber: {AMBER};
  --risk-soft: #F7E6E3; --clear-soft: #E3F1E9; --amber-soft: #F8EDD7; --teal-soft: #E1F0EE;
}}

html, body, [class*="st-"], [data-testid="stAppViewContainer"], button, input, select, textarea {{
  font-family: {FONT};
}}
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] * {{ font-family: "Material Symbols Rounded" !important; }}
[data-testid="stAppViewContainer"] {{ background: var(--canvas); color: var(--ink); }}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{ background: var(--ink); border-right: none; }}
[data-testid="stSidebar"] * {{ color: #C9D6DD; }}
[data-testid="stSidebarNav"]::before {{
  content: "Multi-Disease Prediction";
  display: block; text-align: left; padding: 1.35rem 1.1rem 0.1rem 4rem; font-size: 1.02rem; font-weight: 600; color: #FFFFFF;
  background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E%3Crect width='40' height='40' rx='9' fill='%230B7A75'/%3E%3Cpolyline points='6,22 14,22 18,11 23,31 27,17 30,22 34,22' fill='none' stroke='white' stroke-width='2.6' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") no-repeat 1.1rem 1.15rem / 32px 32px; min-height: 1.6rem;
}}
[data-testid="stSidebarNav"] a {{ border-radius: 6px; margin: 1px 0.5rem; }}
[data-testid="stSidebarNav"] a:hover {{ background: rgba(255,255,255,0.07); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: rgba(255,255,255,0.12); }}
[data-testid="stSidebarNav"] a[aria-current="page"] * {{ color: #FFFFFF; font-weight: 600; }}
.side-foot {{ font-size: 0.78rem; line-height: 1.5; color: #8DA3AF; padding: 0 0.6rem; }}
.side-foot b {{ color: #E4ECF0; font-weight: 600; }}

/* ---- page header ---- */
.page-head {{ margin-bottom: 1.5rem; padding-bottom: 1.2rem; border-bottom: 1px solid var(--line); }}
.chips {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.9rem; }}
.chip {{ background: #FFFFFF; border: 1px solid var(--line); border-radius: 999px; padding: 0.2rem 0.75rem; font-size: 0.8rem; color: var(--ink-2); }}
.chip b {{ color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }}
.page-head h1 {{ font-size: 1.85rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 0.25rem; padding: 0; color: var(--ink); }}
.page-head p {{ margin: 0; color: var(--ink-2); font-size: 0.98rem; max-width: 70ch; line-height: 1.5; }}

/* ---- generic blocks ---- */
.panel {{ background: #FFFFFF; border: 1px solid var(--line); border-radius: 8px; padding: 1.25rem 1.4rem; }}
.panel-title {{ font-size: 1rem; font-weight: 600; margin: 0 0 0.15rem; color: var(--ink); }}
.panel-sub {{ font-size: 0.86rem; color: var(--ink-2); margin: 0 0 0.9rem; line-height: 1.45; }}
.group-title {{ font-size: 0.92rem; font-weight: 600; margin: 1.1rem 0 0.2rem; color: var(--ink); }}
.group-title:first-child {{ margin-top: 0.2rem; }}
.muted {{ color: var(--ink-2); font-size: 0.86rem; line-height: 1.5; }}

/* ---- KPI strip ---- */
.kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); background: #FFFFFF; border: 1px solid var(--line); border-radius: 8px; }}
.kpi {{ padding: 1.25rem 1.5rem 1.35rem; border-right: 1px solid var(--line); }}
.kpi:last-child {{ border-right: none; }}
.kpi .name {{ font-size: 0.95rem; font-weight: 600; color: var(--ink); }}
.kpi .acc {{ font-size: 2.6rem; font-weight: 500; letter-spacing: -0.02em; line-height: 1.15; margin: 0.35rem 0 0.1rem; font-variant-numeric: tabular-nums; }}
.kpi .acc small {{ font-size: 1.2rem; color: var(--ink-2); font-weight: 400; margin-left: 2px; }}
.kpi .cap {{ font-size: 0.82rem; color: var(--ink-2); margin-bottom: 0.8rem; }}
.kpi dl {{ margin: 0; display: grid; grid-template-columns: auto auto; gap: 0.18rem 1rem; font-size: 0.84rem; }}
.kpi dt {{ color: var(--ink-2); }}
.kpi dd {{ margin: 0; text-align: right; font-variant-numeric: tabular-nums; font-weight: 500; }}
.bar-dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; }}

/* ---- pipeline ---- */
.steps {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 0; background: #FFFFFF; border: 1px solid var(--line); border-radius: 8px; }}
.step {{ padding: 1rem 1.15rem 1.1rem; border-right: 1px solid var(--line); }}
.step:last-child {{ border-right: none; }}
.step .n {{ font-size: 0.8rem; color: var(--teal); font-weight: 600; }}
.step .t {{ font-size: 0.95rem; font-weight: 600; margin: 0.1rem 0 0.25rem; }}
.step .d {{ font-size: 0.82rem; color: var(--ink-2); line-height: 1.45; }}

/* ---- result panel ---- */
.result {{ background: #FFFFFF; border: 1px solid var(--line); border-radius: 8px; padding: 1.4rem 1.5rem 1.5rem; }}
.verdict {{ display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; }}
.verdict .v {{ font-size: 1.3rem; font-weight: 600; letter-spacing: -0.01em; line-height: 1.25; }}
.verdict .p {{ font-size: 2.7rem; font-weight: 500; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; line-height: 1; white-space: nowrap; }}
.verdict .p small {{ font-size: 1.1rem; font-weight: 400; color: var(--ink-2); }}
.is-risk .v, .is-risk .p {{ color: var(--risk); }}
.is-clear .v, .is-clear .p {{ color: var(--clear); }}
.is-unsure .v, .is-unsure .p {{ color: var(--amber); }}
.result .sub {{ color: var(--ink-2); font-size: 0.88rem; margin: 0.45rem 0 1.4rem; line-height: 1.5; }}

.ruler {{ position: relative; margin: 3.1rem 0 1.5rem; }}
.zones {{ display: flex; height: 12px; border-radius: 6px; overflow: hidden; }}
.zones span {{ display: block; height: 100%; }}
.z-clear {{ background: #BFDCCB; width: 40%; }}
.z-unsure {{ background: #EBD6A6; width: 20%; }}
.z-risk {{ background: #E3B5AE; width: 40%; }}
.marker {{ position: absolute; top: -9px; width: 4px; height: 30px; background: var(--ink); border-radius: 2px; transform: translateX(-50%); }}
.marker::before {{ content: attr(data-label); position: absolute; bottom: 34px; left: 50%; transform: translateX(-50%);
  background: var(--ink); color: #FFFFFF; font-size: 0.78rem; font-weight: 600; padding: 2px 7px; border-radius: 4px; white-space: nowrap; font-variant-numeric: tabular-nums; }}
.ticks {{ display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--ink-2); margin-top: 0.5rem; font-variant-numeric: tabular-nums; }}
.legend-row {{ display: flex; gap: 1.1rem; flex-wrap: wrap; font-size: 0.8rem; color: var(--ink-2); }}
.legend-row i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}

.notice {{ border-left: 3px solid var(--amber); background: var(--amber-soft); padding: 0.7rem 1rem; border-radius: 0 6px 6px 0; font-size: 0.86rem; line-height: 1.5; color: var(--ink); }}

/* ---- metric rows ---- */
.mrow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0; border: 1px solid var(--line); border-radius: 8px; background: #FFFFFF; }}
.m {{ padding: 0.95rem 1.2rem; border-right: 1px solid var(--line); }}
.m:last-child {{ border-right: none; }}
.m .l {{ font-size: 0.82rem; color: var(--ink-2); }}
.m .x {{ font-size: 1.7rem; font-weight: 500; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }}

/* ---- native widgets ---- */
[data-testid="stWidgetLabel"] p {{ font-size: 0.82rem; color: var(--ink-2); }}
[data-testid="stNumberInput"] input, [data-baseweb="select"] > div {{ background: #FFFFFF; border-radius: 6px; }}
div[data-testid="stButton"] button {{ border-radius: 6px; border: 1px solid var(--line); background: #FFFFFF; color: var(--ink); font-weight: 500; font-size: 0.86rem; }}
div[data-testid="stButton"] button {{ min-height: 2.6rem; height: auto; padding: 0.3rem 0.7rem; }}
div[data-testid="stButton"] button p {{ white-space: normal; line-height: 1.25; text-align: center; }}
div[data-testid="stButton"] button:hover {{ border-color: var(--teal); color: var(--teal); }}
[data-testid="stExpander"] {{ background: #FFFFFF; border: 1px solid var(--line); border-radius: 8px; }}
[data-testid="stExpander"] summary p {{ font-weight: 600; font-size: 0.92rem; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: 8px; }}
button:focus-visible, input:focus-visible {{ outline: 2px solid var(--teal); outline-offset: 2px; }}

/* ---- cards (containers created with key="card_...") ---- */
[class*="st-key-card"] {{ background: #FFFFFF; border: 1px solid var(--line); border-radius: 10px;
  box-shadow: 0 1px 2px rgba(14, 34, 48, 0.05); padding: 1.1rem 1.3rem 1.2rem; }}
[class*="st-key-card"] .group-title {{ margin-top: 0; }}
[data-testid="stPageLink"] a {{ border: 1px solid var(--line); background: #FFFFFF; border-radius: 6px; padding: 0.5rem 0.9rem; }}
[data-testid="stPageLink"] a:hover {{ border-color: var(--teal); }}
[data-testid="stPageLink"] p {{ font-weight: 500; font-size: 0.88rem; }}
.badge {{ display: inline-block; border-radius: 999px; padding: 0.16rem 0.75rem; font-size: 0.78rem; font-weight: 600; margin-bottom: 0.8rem; }}
.is-risk .badge {{ background: var(--risk-soft); color: var(--risk); }}
.is-clear .badge {{ background: var(--clear-soft); color: var(--clear); }}
.is-unsure .badge {{ background: var(--amber-soft); color: var(--amber); }}
.guide {{ margin-top: 1.3rem; padding-top: 0.95rem; border-top: 1px solid var(--line); font-size: 0.86rem; color: var(--ink-2); line-height: 1.55; }}
.fx-label {{ font-size: 0.84rem; color: var(--ink-2); margin-top: 0.5rem; }}
.fx {{ display: inline-block; border-radius: 6px; padding: 0.18rem 0.6rem; margin: 0.25rem 0.35rem 0.1rem 0; font-size: 0.84rem; font-weight: 500; }}
.fx-up {{ background: var(--risk-soft); color: var(--risk); }}
.fx-down {{ background: var(--clear-soft); color: var(--clear); }}
.fx-none {{ color: var(--ink-2); font-size: 0.84rem; margin: 0.25rem 0; }}
.footer {{ margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line); font-size: 0.78rem; color: var(--ink-2); line-height: 1.6; }}

@media (max-width: 900px) {{
  .kpis, .steps, .mrow {{ grid-template-columns: 1fr; }}
  .kpi, .step, .m {{ border-right: none; border-bottom: 1px solid var(--line); }}
  .verdict {{ flex-direction: column; gap: 0.4rem; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(s: str) -> None:
    """Render an HTML snippet (leading whitespace stripped so Markdown never treats it as code)."""
    st.markdown("\n".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def page_header(title: str, subtitle: str, chips=None) -> None:
    chip_html = ""
    if chips:
        chip_html = '<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>"
    html(f'<div class="page-head"><h1>{title}</h1><p>{subtitle}</p>{chip_html}</div>')


# ---------------------------------------------------------------------------
# Disease configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "diabetes": {
        "title": "Diabetes", "noun": "diabetes", "file": "diabetes.csv", "target": "Outcome",
        "pos": "With diabetes", "neg": "Without diabetes", "color": "#2F5D8A",
        "dataset": "Pima Indians Diabetes Database", "icon": ":material/bloodtype:",
        "prep": "Zeros in glucose, blood pressure, skin thickness, insulin and BMI are treated as missing and replaced with the median.",
    },
    "heart": {
        "title": "Heart disease", "noun": "heart disease", "file": "heart.csv", "target": "target",
        "pos": "With heart disease", "neg": "Without heart disease", "color": "#0B7A75",
        "dataset": "Cleveland Heart Disease (UCI)", "icon": ":material/favorite:",
        "prep": "Duplicate rows and rows with missing values are removed before training.",
    },
    "parkinsons": {
        "title": "Parkinson's disease", "noun": "Parkinson's disease", "file": "parkinsons.csv", "target": "status",
        "pos": "With Parkinson's", "neg": "Healthy", "color": "#7A5C99",
        "dataset": "Oxford Parkinson's Disease Detection (UCI)", "icon": ":material/graphic_eq:",
        "prep": "The recording identifier column is dropped; the 22 voice measurements are used as features.",
    },
}

HEART_NOTE = (
    '<div class="muted" style="margin:0.2rem 0 0.9rem"><b>Data note.</b> In this copy of the Cleveland dataset, records labelled 1 '
    '("with heart disease") have higher average maximum heart rate, less ST depression and fewer blocked vessels than records labelled 0. '
    'That pattern usually points the other way, so check what the label means in the original UCI source before reading the direction '
    'of this result as clinical.</div>'
)

# Input specifications. kind: int | float | select
DIABETES_SPECS = [
    dict(name="Pregnancies", label="Pregnancies", group="History", kind="int", lo=0, hi=20, help="Number of times pregnant."),
    dict(name="Age", label="Age", unit="years", group="History", kind="int", lo=18, hi=100),
    dict(name="Glucose", label="Glucose", unit="mg/dL", group="Metabolic markers", kind="int", lo=40, hi=300,
         help="Plasma glucose concentration after a 2-hour oral glucose tolerance test."),
    dict(name="Insulin", label="Insulin", unit="µU/mL", group="Metabolic markers", kind="int", lo=10, hi=900,
         help="2-hour serum insulin."),
    dict(name="BMI", label="BMI", unit="kg/m²", group="Metabolic markers", kind="float", lo=10.0, hi=70.0, step=0.1,
         help="Body mass index."),
    dict(name="BloodPressure", label="Blood pressure", unit="mmHg", group="Vitals and family history", kind="int", lo=20, hi=140,
         help="Diastolic blood pressure."),
    dict(name="SkinThickness", label="Skin thickness", unit="mm", group="Vitals and family history", kind="int", lo=5, hi=100,
         help="Triceps skin fold thickness."),
    dict(name="DiabetesPedigreeFunction", label="Pedigree function", group="Vitals and family history", kind="float",
         lo=0.05, hi=2.5, step=0.01, help="Score estimating hereditary diabetes risk from family history."),
]
DIABETES_ORDER = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

HEART_SPECS = [
    dict(name="age", label="Age", unit="years", group="Patient", kind="int", lo=18, hi=100),
    dict(name="sex", label="Sex", group="Patient", kind="select", options={0: "Female", 1: "Male"}),
    dict(name="cp", label="Chest pain type", group="Symptoms and exercise test", kind="select",
         options={0: "Code 0", 1: "Code 1", 2: "Code 2", 3: "Code 3"}, help="Chest pain type, coded 0 to 3 as in the dataset."),
    dict(name="exang", label="Exercise-induced angina", group="Symptoms and exercise test", kind="select", options={0: "No", 1: "Yes"}),
    dict(name="thalach", label="Maximum heart rate", unit="bpm", group="Symptoms and exercise test", kind="int", lo=60, hi=220),
    dict(name="oldpeak", label="ST depression", group="Symptoms and exercise test", kind="float", lo=0.0, hi=7.0, step=0.1,
         help="ST depression induced by exercise relative to rest."),
    dict(name="slope", label="ST segment slope", group="Symptoms and exercise test", kind="select",
         options={0: "Code 0", 1: "Code 1", 2: "Code 2"}, help="Slope of the peak exercise ST segment, coded 0 to 2."),
    dict(name="trestbps", label="Resting blood pressure", unit="mmHg", group="Clinical tests", kind="int", lo=80, hi=220),
    dict(name="chol", label="Cholesterol", unit="mg/dL", group="Clinical tests", kind="int", lo=100, hi=600),
    dict(name="fbs", label="Fasting blood sugar above 120 mg/dL", group="Clinical tests", kind="select", options={0: "No", 1: "Yes"}),
    dict(name="restecg", label="Resting ECG result", group="Clinical tests", kind="select",
         options={0: "Code 0", 1: "Code 1", 2: "Code 2"}, help="Resting electrocardiogram result, coded 0 to 2."),
    dict(name="ca", label="Major vessels coloured", group="Clinical tests", kind="select",
         options={0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}, help="Number of major vessels coloured by fluoroscopy."),
    dict(name="thal", label="Thalassemia result", group="Clinical tests", kind="select",
         options={0: "Code 0", 1: "Code 1", 2: "Code 2", 3: "Code 3"}, help="Thalassemia test result, coded 0 to 3 as in the dataset."),
]

PARKINSONS_GROUPS = {
    "Fundamental frequency": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"],
    "Frequency variation (jitter)": ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP"],
    "Amplitude variation (shimmer)": ["MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA"],
    "Noise ratios": ["NHR", "HNR"],
    "Nonlinear voice measures": ["RPDE", "DFA", "spread1", "spread2", "D2", "PPE"],
}
PARKINSONS_HELP = {
    "MDVP:Fo(Hz)": "Average vocal fundamental frequency.", "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency.",
    "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency.", "NHR": "Noise-to-harmonics ratio.",
    "HNR": "Harmonics-to-noise ratio. Lower values mean a noisier voice.",
}


# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pickle(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data(key):
    """Dataset prepared exactly as in the matching train_*.py script."""
    path = os.path.join(DATA_DIR, CONFIG[key]["file"])
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if key == "diabetes":
        for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
    elif key == "heart":
        df = df.drop_duplicates().dropna()
    else:
        if "name" in df.columns:
            df = df.drop("name", axis=1)
        df = df.dropna()
    return df


def bundle(key):
    model = load_pickle(f"{key}_model.pkl")
    scaler = load_pickle(f"{key}_scaler.pkl")
    metrics = load_pickle(f"{key}_metrics.pkl")
    if model is None or scaler is None:
        return None
    return dict(model=model, scaler=scaler, metrics=metrics or {})


def feature_order(key, b, df):
    if key == "diabetes":
        return DIABETES_ORDER
    fo = b["metrics"].get("feature_order")
    if fo:
        return list(fo)
    if df is not None:
        return [c for c in df.columns if c != CONFIG[key]["target"]]
    return []


@st.cache_data(show_spinner=False)
def evaluate(key):
    """Recompute test-set metrics with the same split as the training scripts (80/20, seed 42, stratified)."""
    b = bundle(key)
    df = load_data(key)
    if b is None or df is None:
        return None
    target = CONFIG[key]["target"]
    order = feature_order(key, b, df)
    X, y = df[order], df[target]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pred = b["model"].predict(b["scaler"].transform(X_te))
    acc = float((pred == y_te.values).mean())
    stored = b["metrics"].get("accuracy")
    if stored is not None and abs(acc - stored) > 1e-6:
        return dict(acc=float(stored), detail=False, n_test=int(len(y_te)))
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
    cm_ = confusion_matrix(y_te, pred)
    tn_, fp_ = int(cm_[0][0]), int(cm_[0][1])
    return dict(acc=acc, precision=float(p), recall=float(r), f1=float(f), cm=cm_.tolist(),
                specificity=(tn_ / (tn_ + fp_)) if (tn_ + fp_) else 0.0,
                n_test=int(len(y_te)), detail=True)


@st.cache_data(show_spinner=False)
def reference_stats(key):
    """Per-class means/medians and ranges, used for sample patients and comparison charts."""
    df = load_data(key)
    if df is None:
        return None
    target = CONFIG[key]["target"]
    feats = [c for c in df.columns if c != target]
    g0, g1 = df[df[target] == 0][feats], df[df[target] == 1][feats]
    pooled = np.sqrt((g0.var() + g1.var()) / 2).replace(0, np.nan)
    effect = ((g1.mean() - g0.mean()).abs() / pooled).fillna(0)
    return dict(mean0=g0.mean(), mean1=g1.mean(), med0=g0.median(), med1=g1.median(),
                mode0=g0.mode().iloc[0], mode1=g1.mode().iloc[0],
                lo=df[feats].min(), hi=df[feats].max(), effect=effect, n=len(df), n0=len(g0), n1=len(g1))


def plot_style(fig, height=320):
    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=6, b=4), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12.5, color=INK), hoverlabel=dict(font_family=FONT, bgcolor="#FFFFFF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12)),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(color=INK_2))
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(color=INK_2))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def missing_notice(key):
    st.error(f"The {CONFIG[key]['title'].lower()} model files were not found in the models folder. "
             f"Run notebooks/train_{key}.py from inside the notebooks folder to create them.")


# ---------------------------------------------------------------------------
# Prediction page (shared by the three diseases)
# ---------------------------------------------------------------------------
def build_specs(key, df, stats):
    if key == "diabetes":
        return DIABETES_SPECS
    if key == "heart":
        return HEART_SPECS
    specs = []
    for group, names in PARKINSONS_GROUPS.items():
        for n in names:
            if stats is None or n not in stats["lo"].index:
                continue
            lo, hi = float(stats["lo"][n]), float(stats["hi"][n])
            rng = max(hi - lo, 1e-9)
            wlo = lo - 0.25 * rng
            if lo >= 0:
                wlo = max(wlo, 0.0)
            specs.append(dict(name=n, label=n, group=group, kind="float", lo=wlo, hi=hi + 0.25 * rng,
                              step=max(rng / 200, 1e-6), fmt="%.6f", help=PARKINSONS_HELP.get(n)))
    return specs


def coerce(spec, value):
    """Convert a dataset value into something the matching widget accepts."""
    if spec["kind"] == "select":
        opts = list(spec["options"].keys())
        v = int(round(float(value)))
        return v if v in opts else min(opts, key=lambda o: abs(o - v))
    if spec["kind"] == "int":
        return int(min(max(round(float(value)), spec["lo"]), spec["hi"]))
    return float(min(max(float(value), spec["lo"]), spec["hi"]))


@st.cache_data(show_spinner=False)
def representative_records(key):
    """One real record per class: correctly classified by the model and closest to the class median."""
    b, df = bundle(key), load_data(key)
    if b is None or df is None:
        return None
    target = CONFIG[key]["target"]
    order = feature_order(key, b, df)
    X, y = df[order], df[target]
    proba = b["model"].predict_proba(b["scaler"].transform(X))[:, 1]
    sd = X.std().replace(0, 1)
    out = {}
    for cls in (0, 1):
        ok = (y == cls).values & ((proba >= 0.5) == (cls == 1))
        pool = X[ok]
        if pool.empty:
            pool = X[(y == cls).values]
        dist = (((pool - X[y == cls].median()) / sd) ** 2).sum(axis=1)
        out[cls] = pool.loc[dist.idxmin()].to_dict()
    return out


def sample_values(specs, record):
    out = {}
    for s in specs:
        n = s["name"]
        out[n] = coerce(s, record[n] if record and n in record else s.get("lo", 0))
    return out


def apply_preset(key, values):
    for name, v in values.items():
        st.session_state[f"{key}:{name}"] = v


def render_input(key, spec):
    k = f"{key}:{spec['name']}"
    label = f"{spec['label']} ({spec['unit']})" if spec.get("unit") else spec["label"]
    if spec["kind"] == "select":
        opts = list(spec["options"].keys())
        return st.selectbox(label, opts, key=k, format_func=lambda o: spec["options"][o], help=spec.get("help"))
    if spec["kind"] == "int":
        return st.number_input(label, min_value=int(spec["lo"]), max_value=int(spec["hi"]), step=1, key=k, help=spec.get("help"))
    return st.number_input(label, min_value=float(spec["lo"]), max_value=float(spec["hi"]), step=float(spec["step"]),
                           format=spec.get("fmt", "%.2f"), key=k, help=spec.get("help"))


def result_panel(key, b, order, values, ev):
    cfg = CONFIG[key]
    row = pd.DataFrame([[values[n] for n in order]], columns=order)
    scaled = b["scaler"].transform(row)
    pred = int(b["model"].predict(scaled)[0])
    p = float(b["model"].predict_proba(scaled)[0][1])
    pct = p * 100

    if 0.4 <= p <= 0.6:
        state, head, badge = "unsure", f"Close to the decision boundary for {cfg['noun']}", "Uncertain"
        guide = "The model cannot separate this case clearly. Treat it as inconclusive and rely on clinical tests."
    elif pred == 1:
        state, head, badge = "risk", f"Higher likelihood of {cfg['noun']}", "Elevated risk"
        guide = "This is a screening estimate, not a diagnosis. A clinician would confirm it with laboratory or clinical tests."
    else:
        state, head, badge = "clear", f"Lower likelihood of {cfg['noun']}", "Lower risk"
        guide = "A lower estimate does not rule the condition out. Regular check-ups still matter."

    acc_txt = f"{ev['acc'] * 100:.1f}%" if ev else "n/a"
    html(f"""
    <div class="result is-{state}">
      <div class="badge">{badge}</div>
      <div class="verdict"><div class="v">{head}</div><div class="p">{pct:.0f}<small>%</small></div></div>
      <div class="sub">Estimated probability of {cfg['noun']} from the SVM model. On held-out records this model was correct {acc_txt} of the time.</div>
      <div class="ruler">
        <div class="zones"><span class="z-clear"></span><span class="z-unsure"></span><span class="z-risk"></span></div>
        <div class="marker" style="left:{min(max(pct, 1), 99):.1f}%" data-label="{pct:.0f}%"></div>
        <div class="ticks"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
      </div>
      <div class="legend-row">
        <span><i style="background:#BFDCCB"></i>Lower (below 40%)</span>
        <span><i style="background:#EBD6A6"></i>Uncertain (40 to 60%)</span>
        <span><i style="background:#E3B5AE"></i>Higher (above 60%)</span>
      </div>
      <div class="guide">{guide}</div>
    </div>
    """)
    return pct, head


def key_factors(order, values, stats, labels):
    """Which entered values sit closer to the typical record of each group (descriptive, not causal)."""
    lo, hi = stats["lo"], stats["hi"]
    rows = []
    for f in order:
        rng = max(float(hi[f] - lo[f]), 1e-12)
        m0, m1 = (stats["mean0"][f] - lo[f]) / rng, (stats["mean1"][f] - lo[f]) / rng
        gap = m1 - m0
        eff = float(stats["effect"][f])
        if abs(gap) < 1e-9 or eff < 0.3:
            continue
        v = (values[f] - lo[f]) / rng
        score = (v - (m0 + m1) / 2) / (gap / 2)  # about -1 at the typical record without, +1 with the condition
        rows.append((labels.get(f, f), score, eff))
    toward = sorted([r for r in rows if r[1] > 0.4], key=lambda r: r[1] * r[2], reverse=True)[:3]
    away = sorted([r for r in rows if r[1] < -0.4], key=lambda r: -r[1] * r[2], reverse=True)[:3]
    return [r[0] for r in toward], [r[0] for r in away]


def result_csv(key, order, values, pct, head, labels):
    cfg = CONFIG[key]
    rows = [("Report", f"{cfg['title']} screening estimate"), ("Result", head), ("Estimated probability (%)", f"{pct:.1f}"),
            ("Model", "Support Vector Machine, RBF kernel")]
    rows += [(labels.get(f, f), values[f]) for f in order]
    rows.append(("Note", "Educational tool. Not a medical device and not a diagnosis."))
    return pd.DataFrame(rows, columns=["Field", "Value"]).to_csv(index=False).encode("utf-8")


def comparison_chart(key, order, values, stats, labels):
    """Where this patient sits between typical records without and with the condition."""
    cfg = CONFIG[key]
    if stats is None:
        return
    top = stats["effect"].reindex(order).sort_values(ascending=False).head(6).index.tolist()
    top = top[::-1]  # most distinctive feature at the top of the chart
    ylab = {f: labels.get(f, f) for f in top}
    lo, hi = stats["lo"], stats["hi"]

    def norm(feat, v):
        rng = max(float(hi[feat] - lo[feat]), 1e-12)
        return float(min(max((v - lo[feat]) / rng, 0), 1))

    fig = go.Figure()
    for feat in top:
        fig.add_trace(go.Scatter(x=[norm(feat, stats["mean0"][feat]), norm(feat, stats["mean1"][feat])], y=[ylab[feat]] * 2,
                                 mode="lines", line=dict(color=LINE, width=4), hoverinfo="skip", showlegend=False))

    def series(name, xs, text, **marker):
        fig.add_trace(go.Scatter(x=xs, y=[ylab[f] for f in top], mode="markers", name=name, marker=marker, text=text,
                                 hovertemplate="%{y}<br>%{text}<extra>" + name + "</extra>"))

    series(cfg["neg"], [norm(f, stats["mean0"][f]) for f in top], [f"Typical value {stats['mean0'][f]:.4g}" for f in top],
           size=12, color=CLEAR)
    series(cfg["pos"], [norm(f, stats["mean1"][f]) for f in top], [f"Typical value {stats['mean1'][f]:.4g}" for f in top],
           size=12, color=RISK)
    series("This patient", [norm(f, values[f]) for f in top], [f"Entered value {values[f]:.4g}" for f in top],
           size=15, color=INK, symbol="diamond", line=dict(color="#FFFFFF", width=2))
    plot_style(fig, 300)
    fig.update_xaxes(range=[-0.04, 1.04], tickvals=[0, 0.5, 1], ticktext=["Low", "Mid", "High"], title_text="Position within the dataset range", title_font=dict(color=INK_2))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    show(fig)


NOTICE_TEXT = ("<b>Clinical decision-support prototype.</b> Not validated on local patient data and not approved as a medical device. "
               "Estimates must be reviewed by a qualified clinician and never used alone for diagnosis or treatment.")


def out_of_range(specs, values, stats):
    """Numeric inputs outside the range of the training data, where the model has seen nothing similar."""
    flagged = []
    if not stats:
        return flagged
    for sp in specs:
        n = sp["name"]
        if sp["kind"] == "select" or n not in stats["lo"].index:
            continue
        if values[n] < float(stats["lo"][n]) or values[n] > float(stats["hi"][n]):
            flagged.append(sp["label"])
    return flagged


def add_to_log(key, head, pct):
    ss = st.session_state
    ss.setdefault("log", []).append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "Patient ID": ss.get("enc_pid", ""), "Clinician": ss.get("enc_clin", ""),
        "Condition": CONFIG[key]["title"], "Probability (%)": round(pct, 1), "Result": head,
    })
    st.toast("Added to the session log")


def report_html(key, specs, values, pct, head, ev, flagged):
    cfg, ss = CONFIG[key], st.session_state
    rows = ""
    for sp in specs:
        v = values[sp["name"]]
        shown = sp["options"][v] if sp["kind"] == "select" else (f"{v:g}" if isinstance(v, float) else v)
        unit = f" {esc(sp['unit'])}" if sp.get("unit") else ""
        rows += f"<tr><td>{esc(sp['label'])}</td><td>{esc(str(shown))}{unit}</td></tr>"
    perf = ""
    if ev and ev.get("detail"):
        perf = (f"Held-out test accuracy {ev['acc']*100:.1f}%, sensitivity {ev['recall']*100:.1f}%, specificity {ev['specificity']*100:.1f}%, "
                f"precision {ev['precision']*100:.1f}% on {ev['n_test']} test records.")
    warn = (f"<p class='warn'>Values outside the range of the training data: {esc(', '.join(flagged))}. The estimate may be unreliable.</p>"
            if flagged else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(cfg['title'])} screening report</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;color:#0E2230;margin:0;padding:32px 40px;max-width:780px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:14px;margin:22px 0 6px;border-bottom:1px solid #D9E1E4;padding-bottom:4px}}
.sub{{color:#506676;font-size:13px}} table{{border-collapse:collapse;width:100%;font-size:13px}}
td{{border-bottom:1px solid #E6ECED;padding:5px 6px}} td:first-child{{width:55%;color:#506676}}
.result{{border:1px solid #D9E1E4;border-radius:8px;padding:14px 18px;margin-top:10px}}
.big{{font-size:30px;font-weight:600}} .warn{{background:#F8EDD7;border-left:3px solid #B7791F;padding:8px 12px;font-size:13px}}
.small{{font-size:12px;color:#506676;line-height:1.5}} .sign{{display:flex;gap:40px;margin-top:36px;font-size:13px}}
.sign div{{flex:1;border-top:1px solid #0E2230;padding-top:4px}}
@media print{{body{{padding:0}}}}
</style></head><body>
<h1>{esc(cfg['title'])} screening estimate</h1>
<div class="sub">Generated {datetime.now().strftime('%d %b %Y, %H:%M')} by the Multi-Disease Prediction System</div>
<h2>Patient and encounter</h2>
<table><tr><td>Patient ID or MRN</td><td>{esc(ss.get('enc_pid','') or '-')}</td></tr>
<tr><td>Patient name</td><td>{esc(ss.get('enc_name','') or '-')}</td></tr>
<tr><td>Clinician</td><td>{esc(ss.get('enc_clin','') or '-')}</td></tr></table>
<h2>Model estimate</h2>
<div class="result"><div class="big">{pct:.0f}%</div><div>{esc(head)}</div></div>{warn}
<h2>Measurements entered</h2><table>{rows}</table>
<h2>Model and limitations</h2>
<p class="small">Support Vector Machine (RBF kernel) trained on the {esc(cfg['dataset'])}. {esc(perf)}
This is a decision-support prototype. It has not been validated on local patient data and is not approved as a medical device.
The estimate describes similarity to historical records and does not confirm or exclude a diagnosis.</p>
<div class="sign"><div>Reviewed by (name and signature)</div><div>Date</div></div>
</body></html>""".encode("utf-8")


def enc_field(label, store_key):
    """Text input whose value survives page changes (Streamlit drops a widget's state when its page is not shown)."""
    ss = st.session_state
    ss.setdefault(store_key, "")

    def _sync():
        ss[store_key] = ss[f"w_{store_key}"]

    return st.text_input(label, value=ss[store_key], key=f"w_{store_key}", on_change=_sync)


def log_page():
    page_header("Session log", "Estimates added during this session. They are kept in memory only and are not written to disk by this app.")
    rows = st.session_state.get("log", [])
    if not rows:
        html('<div class="notice">Nothing logged yet. On a prediction page, press <b>Add to log</b> to record an estimate here.</div>')
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("Download log as CSV", df.to_csv(index=False).encode("utf-8"), "session_log.csv", "text/csv", width="stretch")
    if c2.button("Clear log", width="stretch"):
        st.session_state["log"] = []
        st.rerun()


def predictor_page(key):
    cfg = CONFIG[key]
    with st.spinner("Loading model and dataset..."):
        ev0 = evaluate(key)
        stats0 = reference_stats(key)
        representative_records(key)
    chips = ["<b>SVM</b> with RBF kernel"]
    if ev0:
        chips.append(f"Test accuracy <b>{ev0['acc'] * 100:.1f}%</b>")
    if stats0:
        chips.append(f"<b>{stats0['n']:,}</b> training records")
    page_header(f"{cfg['title']} risk prediction",
                f"Enter patient measurements to see the model's estimate. Results update as you change a value. Dataset: {cfg['dataset']}.",
                chips)
    b = bundle(key)
    if b is None:
        missing_notice(key)
        return
    df = load_data(key)
    stats = reference_stats(key)
    ev = evaluate(key)
    specs = build_specs(key, df, stats)
    order = feature_order(key, b, df)
    if not specs or not order:
        st.error(f"The {cfg['title'].lower()} dataset was not found in the data folder, so the input form cannot be built.")
        return

    # initialise widget state once (typical record without the condition)
    recs = representative_records(key)
    base = sample_values(specs, recs[0] if recs else None)
    for s in specs:
        st.session_state.setdefault(f"{key}:{s['name']}", base[s["name"]])

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        with st.container(key=f"card_encounter_{key}"):
            html('<div class="group-title">Patient and encounter</div>'
                 '<div class="muted" style="margin-bottom:0.6rem">Kept in memory for this session only. The app does not write these details to disk.</div>')
            e1, e2 = st.columns(2)
            with e1:
                enc_field("Patient ID or MRN", "enc_pid")
            with e2:
                enc_field("Patient name (optional)", "enc_name")
            enc_field("Clinician", "enc_clin")
        with st.container(key=f"card_presets_{key}"):
            html('<div class="group-title">Patient record</div>'
                 '<div class="muted" style="margin-bottom:0.6rem">Start from a real record from the dataset (the model classifies each correctly), then adjust any value.</div>')
            c1, c2 = st.columns(2)
            c1.button("Low-risk sample", key=f"{key}-neg", width="stretch",
                      on_click=apply_preset, args=(key, base))
            c2.button("High-risk sample", key=f"{key}-pos", width="stretch",
                      on_click=apply_preset, args=(key, sample_values(specs, recs[1] if recs else None)))

        groups = {}
        for s in specs:
            groups.setdefault(s["group"], []).append(s)

        values = {}
        if key == "parkinsons":
            for i, (g, items) in enumerate(groups.items()):
                with st.expander(g, expanded=(i == 0)):
                    cols = st.columns(2)
                    for j, s in enumerate(items):
                        with cols[j % 2]:
                            values[s["name"]] = render_input(key, s)
        else:
            for gi, (g, items) in enumerate(groups.items()):
                with st.container(key=f"card_group_{key}_{gi}"):
                    html(f'<div class="group-title">{g}</div>')
                    cols = st.columns(2)
                    for j, s in enumerate(items):
                        with cols[j % 2]:
                            values[s["name"]] = render_input(key, s)

    with right:
        labels = {sp["name"]: sp["label"] for sp in specs}
        pct, head = result_panel(key, b, order, values, ev)
        flagged = out_of_range(specs, values, stats)
        if flagged:
            html(f'<div class="notice" style="margin-top:0.9rem"><b>Check these values.</b> {esc(", ".join(flagged))} '
                 f'{"is" if len(flagged) == 1 else "are"} outside the range of the training data, so the estimate may be unreliable.</div>')
        st.write("")
        with st.container(key=f"card_factors_{key}"):
            html('<div class="panel-title">What stands out</div>'
                 '<div class="panel-sub">Entered values that sit closer to the typical record of one group than the other.</div>')
            up, down = key_factors(order, values, stats, labels) if stats else ([], [])
            up_html = "".join(f'<span class="fx fx-up">{x}</span>' for x in up) or '<div class="fx-none">None stand out</div>'
            down_html = "".join(f'<span class="fx fx-down">{x}</span>' for x in down) or '<div class="fx-none">None stand out</div>'
            html(f'<div class="fx-label">Closer to the group {cfg["pos"][0].lower() + cfg["pos"][1:]}</div>{up_html}'
                 f'<div class="fx-label" style="margin-top:0.8rem">Closer to the group {cfg["neg"][0].lower() + cfg["neg"][1:]}</div>{down_html}')
        st.write("")
        with st.container(key=f"card_compare_{key}"):
            html('<div class="panel-title">How this patient compares</div>'
                 '<div class="panel-sub">The six measurements that best separate the two groups in the training data. '
                 'Dots show the typical value for each group, scaled to the dataset range.</div>')
            comparison_chart(key, order, values, stats, labels)
        st.write("")
        a1, a2, a3 = st.columns(3)
        a1.button("Add to log", key=f"{key}-log", width="stretch", on_click=add_to_log, args=(key, head, pct))
        a2.download_button("Report (HTML)", data=report_html(key, specs, values, pct, head, ev, flagged),
                           file_name=f"{key}_screening_report.html", mime="text/html", width="stretch")
        a3.download_button("CSV", data=result_csv(key, order, values, pct, head, labels),
                           file_name=f"{key}_screening_result.csv", mime="text/csv", width="stretch")
        st.caption("Open the HTML report in a browser and press Ctrl+P to print it or save it as a PDF.")
        if key == "heart":
            html(HEART_NOTE)
        html(f'<div class="notice">{NOTICE_TEXT}</div>')


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def overview_page():
    with st.spinner("Loading models and datasets..."):
        for k in CONFIG:  # first visit only; results are cached afterwards
            evaluate(k)
            reference_stats(k)
    total = sum((reference_stats(k) or {"n": 0})["n"] for k in CONFIG)
    page_header("Multi-disease risk screening",
                "Three Support Vector Machine models estimate the likelihood of diabetes, heart disease and Parkinson's disease "
                "from clinical and voice measurements.",
                ["<b>3</b> trained models", f"<b>{total:,}</b> patient records", "Prototype, not clinically validated"])
    cards, evs = [], {}
    for key, cfg in CONFIG.items():
        ev, st_ = evaluate(key), reference_stats(key)
        evs[key] = ev
        feats = len(st_["lo"]) if st_ else "n/a"
        acc = f"{ev['acc'] * 100:.1f}<small>%</small>" if ev else "n/a"
        recs = f"{st_['n']:,}" if st_ else "n/a"
        ntest = ev["n_test"] if ev else "n/a"
        cards.append(f"""
        <div class="kpi">
          <div class="name"><span class="bar-dot" style="background:{cfg['color']}"></span>{cfg['title']}</div>
          <div class="acc">{acc}</div>
          <div class="cap">Accuracy on held-out records</div>
          <dl><dt>Records</dt><dd>{recs}</dd><dt>Input features</dt><dd>{feats}</dd><dt>Test records</dt><dd>{ntest}</dd></dl>
        </div>""")
    html(f'<div class="kpis">{"".join(cards)}</div>')
    st.write("")
    links = st.columns(3, gap="medium")
    for col, (key, cfg), idx in zip(links, CONFIG.items(), (1, 2, 3)):
        with col:
            st.page_link(pages[idx], label=f"Open {cfg['title']} prediction", icon=cfg["icon"], width="stretch")
    st.write("")

    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        with st.container(key="card_overview_perf"):
            html('<div class="panel-title">Test performance by model</div>'
                 '<div class="panel-sub">Accuracy alone can hide mistakes. Recall is the share of real cases the model catches, '
                 'precision is the share of flagged cases that are real.</div>')
            metrics = [("Accuracy", "acc"), ("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1")]
            fig = go.Figure()
            for key, cfg in CONFIG.items():
                ev = evs[key]
                if ev and ev.get("detail"):
                    fig.add_trace(go.Bar(name=cfg["title"], x=[m[0] for m in metrics], y=[ev[m[1]] * 100 for m in metrics],
                                         marker_color=cfg["color"], hovertemplate="%{x}: %{y:.1f}%<extra>" + cfg["title"] + "</extra>"))
            plot_style(fig, 330)
            fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
            fig.update_yaxes(range=[0, 100], ticksuffix="%")
            fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
            show(fig)
    with c2:
        with st.container(key="card_overview_balance"):
            html('<div class="panel-title">Class balance in each dataset</div>'
                 '<div class="panel-sub">Parkinson\'s records are mostly positive, which is why the models use balanced class weights.</div>')
            fig = go.Figure()
            names, neg, pos = [], [], []
            for key, cfg in CONFIG.items():
                s_ = reference_stats(key)
                if s_:
                    names.append(cfg["title"]); neg.append(s_["n0"]); pos.append(s_["n1"])
            fig.add_trace(go.Bar(y=names, x=neg, name="Without condition", orientation="h", marker_color=CLEAR))
            fig.add_trace(go.Bar(y=names, x=pos, name="With condition", orientation="h", marker_color=RISK))
            plot_style(fig, 330)
            fig.update_layout(barmode="stack", bargap=0.45)
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            show(fig)

    st.write("")
    html('<div class="panel-title">How each model is built</div>'
         '<div class="panel-sub">The same five steps are applied to every dataset.</div>')
    html("""
    <div class="steps">
      <div class="step"><div class="n">1</div><div class="t">Clean</div><div class="d">Fix invalid zeros, drop duplicates and identifier columns.</div></div>
      <div class="step"><div class="n">2</div><div class="t">Split</div><div class="d">80% training and 20% test, stratified so class ratios match.</div></div>
      <div class="step"><div class="n">3</div><div class="t">Scale</div><div class="d">StandardScaler fitted on training data only to avoid leakage.</div></div>
      <div class="step"><div class="n">4</div><div class="t">Tune</div><div class="d">GridSearchCV with 5-fold cross-validation over C, gamma and kernel.</div></div>
      <div class="step"><div class="n">5</div><div class="t">Predict</div><div class="d">Probability estimate from the tuned SVM, served in this app.</div></div>
    </div>
    """)


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
def confusion_chart(cm, cfg):
    labels = [cfg["neg"], cfg["pos"]]
    z = np.array(cm)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Predicted: {l.lower()}" for l in labels], y=[f"Actual: {l.lower()}" for l in labels],
        colorscale=[[0, "#F4F8F8"], [1, TEAL]], showscale=False, hoverinfo="skip",
        text=z.astype(str), texttemplate="%{text}", textfont=dict(size=22, family=FONT),
    ))
    plot_style(fig, 300)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)", side="top")
    return fig


def performance_page():
    page_header("Model performance",
                "Every number on this page is recomputed from the saved models on the same 20% test split used during training.")
    tabs = st.tabs([cfg["title"] for cfg in CONFIG.values()])
    for tab, (key, cfg) in zip(tabs, CONFIG.items()):
        with tab:
            b, ev = bundle(key), evaluate(key)
            if b is None or ev is None:
                missing_notice(key)
                continue
            if ev.get("detail"):
                html(f"""
                <div class="mrow">
                  <div class="m"><div class="l">Accuracy</div><div class="x">{ev['acc']*100:.1f}%</div></div>
                  <div class="m"><div class="l">Precision</div><div class="x">{ev['precision']*100:.1f}%</div></div>
                  <div class="m"><div class="l">Recall (sensitivity)</div><div class="x">{ev['recall']*100:.1f}%</div></div>
                  <div class="m"><div class="l">Specificity</div><div class="x">{ev['specificity']*100:.1f}%</div></div>
                  <div class="m"><div class="l">F1 score</div><div class="x">{ev['f1']*100:.1f}%</div></div>
                </div>""")
            else:
                html(f'<div class="mrow" style="grid-template-columns:1fr"><div class="m"><div class="l">Accuracy</div><div class="x">{ev["acc"]*100:.1f}%</div></div></div>')
            st.write("")
            c1, c2 = st.columns([1.1, 1], gap="large")
            with c1:
                if ev.get("detail"):
                    html('<div class="panel-title">Confusion matrix</div>'
                         f'<div class="panel-sub">{ev["n_test"]} test records. Rows are the true class, columns are the model\'s prediction.</div>')
                    show(confusion_chart(ev["cm"], cfg))
                else:
                    st.info("Detailed metrics are not available because the test split could not be reproduced.")
            with c2:
                params = b["model"].get_params()
                rows = [("Algorithm", "Support Vector Classifier"), ("Kernel", params.get("kernel")), ("C (regularisation)", params.get("C")),
                        ("Gamma", params.get("gamma")), ("Class weight", params.get("class_weight") or "none"),
                        ("Probability output", "Platt scaling" if params.get("probability") else "off"),
                        ("Test records", ev["n_test"])]
                body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
                html(f"""
                <div class="panel-title">Selected hyperparameters</div>
                <div class="panel-sub">Chosen by GridSearchCV with cross-validation on the training data.</div>
                <table style="width:100%;border-collapse:collapse;font-size:0.9rem;background:#fff;border:1px solid {LINE};border-radius:8px">
                {body.replace('<td>', '<td style="padding:0.55rem 0.9rem;border-bottom:1px solid ' + LINE + '">')}
                </table>""")
                st.write("")
                html(f'<div class="muted"><b>Reading this honestly.</b> With only {ev["n_test"]} test records, one extra mistake moves accuracy by '
                     f'about {100/ev["n_test"]:.1f} percentage points, so treat these scores as estimates rather than exact figures.</div>')


# ---------------------------------------------------------------------------
# Data explorer
# ---------------------------------------------------------------------------
def data_page():
    page_header("Data explorer", "Inspect the datasets the models were trained on and how each measurement differs between groups.")
    key = st.segmented_control("Dataset", list(CONFIG.keys()), default="diabetes", format_func=lambda k: CONFIG[k]["title"],
                               label_visibility="collapsed") or "diabetes"
    cfg, df, s = CONFIG[key], load_data(key), reference_stats(key)
    if df is None:
        st.error("Dataset file not found in the data folder.")
        return
    target = cfg["target"]
    html(f"""
    <div class="mrow">
      <div class="m"><div class="l">Records after cleaning</div><div class="x">{s['n']:,}</div></div>
      <div class="m"><div class="l">Input features</div><div class="x">{len(s['lo'])}</div></div>
      <div class="m"><div class="l">{cfg['pos']}</div><div class="x">{s['n1']:,}</div></div>
      <div class="m"><div class="l">{cfg['neg']}</div><div class="x">{s['n0']:,}</div></div>
    </div>""")
    html(f'<div class="muted" style="margin:0.7rem 0 1rem"><b>{cfg["dataset"]}.</b> {cfg["prep"]}</div>')
    if key == "heart":
        html(HEART_NOTE)

    feats = [c for c in df.columns if c != target]
    feat = st.selectbox("Measurement to compare", feats, index=int(np.argmax(s["effect"].reindex(feats).values)))
    fig = go.Figure()
    for cls, name, color in [(0, cfg["neg"], CLEAR), (1, cfg["pos"], RISK)]:
        fig.add_trace(go.Histogram(x=df[df[target] == cls][feat], name=name, marker_color=color, opacity=0.65, nbinsx=24))
    plot_style(fig, 340)
    fig.update_layout(barmode="overlay", bargap=0.04)
    fig.update_xaxes(title_text=feat, title_font=dict(color=INK_2), gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title_text="Records", title_font=dict(color=INK_2))
    show(fig)
    st.caption("The measurement above is pre-selected as the one that separates the two groups most strongly.")
    with st.expander("Preview the first 15 records"):
        st.dataframe(df.head(15), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
def about_page():
    page_header("About this project",
                "A multi-disease prediction system built as the capstone of the Robokwik Prabandhan AI/ML/DS internship (IIT Kanpur).")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        html("""
        <div class="panel">
          <div class="panel-title">What it does</div>
          <div class="panel-sub">Predicts the likelihood of diabetes, heart disease and Parkinson's disease from clinical and voice
          measurements, using one Support Vector Machine per disease with an RBF kernel (selected per model by grid search).</div>
          <div class="panel-title">Technology</div>
          <div class="panel-sub" style="margin-bottom:0">
          Python for the whole pipeline. pandas and NumPy for data handling. scikit-learn for scaling, SVM, grid search and metrics.
          Plotly for charts. Streamlit for the web interface. pickle to save and load trained models.</div>
        </div>""")
    with c2:
        html("""
        <div class="panel">
          <div class="panel-title">Limitations</div>
          <div class="panel-sub" style="margin-bottom:0.9rem">The datasets are small public benchmarks, and the test sets contain only a few dozen
          records, so the scores are rough estimates. The models are not clinically validated and may not generalise to other populations.</div>
          <div class="panel-title">Built by</div>
          <div class="panel-sub" style="margin-bottom:0">Reddi Lahari, B.Tech Computer Science Engineering,
          Andhra University College of Engineering for Women.</div>
        </div>""")
    st.write("")
    html(f'<div class="notice">{NOTICE_TEXT}</div>')


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def diabetes_page(): predictor_page("diabetes")
def heart_page(): predictor_page("heart")
def parkinsons_page(): predictor_page("parkinsons")


pages = [
    st.Page(overview_page, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(diabetes_page, title="Diabetes", icon=":material/bloodtype:", url_path="diabetes"),
    st.Page(heart_page, title="Heart disease", icon=":material/favorite:", url_path="heart"),
    st.Page(parkinsons_page, title="Parkinson's disease", icon=":material/graphic_eq:", url_path="parkinsons"),
    st.Page(log_page, title="Session log", icon=":material/history:", url_path="log"),
    st.Page(performance_page, title="Model performance", icon=":material/monitoring:", url_path="performance"),
    st.Page(data_page, title="Data explorer", icon=":material/database:", url_path="data"),
    st.Page(about_page, title="About", icon=":material/info:", url_path="about"),
]
nav = st.navigation(pages)

with st.sidebar:
    html('<div class="side-foot" style="margin-top:1.5rem"><b>Decision-support prototype.</b><br>Not clinically validated.<br><br>'
         'Built by Lahari Reddi<br>B.Tech CSE, Andhra University</div>')

nav.run()

html("""
<div class="footer"><b>Multi-Disease Prediction System</b><br>
Built by Reddi Lahari, B.Tech CSE, Andhra University College of Engineering for Women, during the Robokwik Prabandhan AI/ML/DS internship (IIT Kanpur).<br>
Decision-support prototype. Not validated for clinical use, not a medical device, and not a substitute for professional medical judgement.</div>
""")
