"""NutriPath - Streamlit app (prototype built on synthetic data).

Tab 1: population analytics dashboard
Tab 2: personal A1C prediction + plain-English plan

Repo layout expected by this file:
    app.py
    requirements.txt
    model/nutripath_xgb.json
    data/dashboard_data.parquet
Secret needed (Streamlit Cloud > Settings > Secrets):
    GROQ_API_KEY = "..."
"""
import warnings
from pathlib import Path

import plotly.express as px
import streamlit as st
from groq import Groq
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
st.set_page_config(page_title="NutriPath", page_icon="🥗", layout="wide")
BASE_DIR = Path(__file__).parent



# ══════════════════════════ Look and feel ════════════════════════════════════
import html as _html

INK, MUTED, LINE = "#16241E", "#5B6B63", "#DCE4DE"
BG, SURFACE = "#F6F8F5", "#FFFFFF"
BRAND, BRAND_DARK, BRAND_TINT, BAR = "#14594A", "#0E4538", "#E3EFE9", "#2B8A6E"
# Risk tiers: one calm blue ramp, darker = higher risk. Every use is also labelled in words.
RISK = {"Low": "#7FA7D8", "Moderate": "#4F7FC0", "High": "#2E5A9E", "Critical": "#183A70"}
ZONE_FILL = {"Low": "#DDE9F7", "Moderate": "#BFD4EF", "High": "#9FBBE2", "Critical": "#7F9FCF"}
FONT_STACK = "Atkinson Hyperlegible, system-ui, -apple-system, Segoe UI, sans-serif"


def h(text):
    """Escape text for our own HTML blocks ($ is escaped so it is never read as math)."""
    return _html.escape(str(text)).replace("$", "&#36;")


THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Young+Serif&display=swap');
.stApp { background: @BG@; color: @INK@; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp [data-baseweb="select"], .stApp [data-testid="stCaptionContainer"] { font-family: @FONT@; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Young Serif', Georgia, serif; font-weight: 400; color: @INK@; letter-spacing: 0; }
.stApp h3 { font-size: 1.55rem; margin-top: 1.6rem; }
.stApp h4 { font-size: 1.2rem; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1120px; padding-top: 1.6rem; padding-bottom: 3rem; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid @LINE@; }
.stTabs [data-baseweb="tab"] { height: 3rem; padding: 0 1.1rem; }
.stTabs [data-baseweb="tab"] p { font-size: 1.05rem; font-weight: 700; color: @MUTED@; }
.stTabs [aria-selected="true"] p { color: @BRAND@; }
.stTabs [data-baseweb="tab-highlight"] { background-color: @BRAND@; height: 3px; }
[data-testid="stForm"] { background: @SURFACE@; border: 1px solid @LINE@; border-radius: 14px; padding: 1.4rem 1.4rem 1.2rem; }
[data-testid="stWidgetLabel"] p { font-size: 0.95rem; font-weight: 700; color: @INK@; }
[data-testid="stFormSubmitButton"] button, .stButton button[kind="primary"] { background: @BRAND@; border: 0; color: #fff; border-radius: 10px; min-height: 3rem; padding: 0 1.8rem; font-weight: 700; font-size: 1.05rem; }
[data-testid="stFormSubmitButton"] button:hover, .stButton button[kind="primary"]:hover { background: @BRAND_DARK@; color: #fff; }
[data-testid="stDownloadButton"] button { background: @SURFACE@; color: @BRAND@; border: 1.5px solid @BRAND@; border-radius: 10px; font-weight: 700; min-height: 2.8rem; }
[data-testid="stDownloadButton"] button:hover { background: @BRAND_TINT@; color: @BRAND_DARK@; border-color: @BRAND_DARK@; }
[data-testid="stExpander"] { background: @SURFACE@; border: 1px solid @LINE@; border-radius: 12px; }
[data-testid="stMetric"] { background: @SURFACE@; border: 1px solid @LINE@; border-radius: 12px; padding: 0.9rem 1.1rem; }
[data-testid="stMetricLabel"] p { color: @MUTED@; font-size: 0.9rem; }
[data-testid="stMetricValue"] { font-family: 'Young Serif', Georgia, serif; }
[data-baseweb="tag"] { background-color: @BRAND_TINT@ !important; }
[data-baseweb="tag"] span { color: @BRAND_DARK@ !important; }
.np-hero { padding: 0.4rem 0 1.2rem; }
.np-brand { display: flex; align-items: center; gap: 0.7rem; }
.np-wordmark { font-family: 'Young Serif', Georgia, serif; font-size: 2.5rem; line-height: 1; color: @INK@; }
.np-tag { margin: 0.7rem 0 0; font-size: 1.12rem; color: @MUTED@; max-width: 44rem; }
.np-panel { background: @SURFACE@; border: 1px solid @LINE@; border-radius: 14px; padding: 1.3rem 1.5rem 1.2rem; margin: 0.6rem 0 0.4rem; }
.stApp .np-ptitle { font-family: 'Young Serif', Georgia, serif; font-size: 1.35rem; margin: 0 0 0.3rem; }
.stApp .np-swaps h4, .stApp .np-two h3 { margin-top: 0; }
.np-ruler { width: 100%; max-width: 760px; height: auto; display: block; margin: 0.2rem 0; }
.np-narrow { display: none; }
.np-stats { display: flex; flex-wrap: wrap; gap: 0.6rem 2.6rem; margin: 0.6rem 0 0.9rem; }
.np-stats span { display: block; font-size: 0.88rem; color: @MUTED@; }
.np-stats b { font-family: 'Young Serif', Georgia, serif; font-weight: 400; font-size: 1.5rem; }
.np-dot { display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 50%; margin-right: 0.45rem; }
.np-snapshot { font-size: 1.1rem; line-height: 1.55; margin: 0; max-width: 46rem; }
.np-safety { background: #FFF3D6; border: 1px solid #E3B84F; border-left: 6px solid #D49A12; border-radius: 12px; padding: 1rem 1.2rem; margin: 0.4rem 0 0.8rem; }
.np-safety b { display: block; margin-bottom: 0.3rem; }
.np-safety ul, .np-list { margin: 0.2rem 0 0; padding-left: 1.2rem; }
.np-safety li, .np-list li { margin: 0.3rem 0; line-height: 1.5; }
.np-note { background: @BRAND_TINT@; border-radius: 12px; padding: 0.9rem 1.2rem; margin: 1.2rem 0 0.8rem; color: @BRAND_DARK@; }
.np-grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.np-card { background: @SURFACE@; border: 1px solid @LINE@; border-radius: 12px; padding: 1.1rem 1.2rem; }
.np-num { font-family: 'Young Serif', Georgia, serif; font-size: 2.1rem; line-height: 1; color: @BRAND@; margin-bottom: 0.5rem; }
.np-action { font-weight: 700; font-size: 1.08rem; line-height: 1.4; }
.np-why { margin-top: 0.6rem; color: @MUTED@; font-size: 0.97rem; line-height: 1.45; }
.np-menu { background: @SURFACE@; border: 1px solid @LINE@; border-radius: 12px; padding: 0.3rem 1.2rem; }
.np-meal { display: grid; grid-template-columns: 7.5rem 1fr; gap: 1rem; padding: 0.85rem 0; border-bottom: 1px solid @LINE@; }
.np-meal:last-child { border-bottom: 0; }
.np-meal span { font-family: 'Young Serif', Georgia, serif; color: @BRAND@; font-size: 1.05rem; }
.np-swaps { margin-top: 1rem; }
.np-swaps h4 { margin: 0 0 0.4rem; }
.np-swap { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.6rem; padding: 0.35rem 0; }
.np-swap .from { color: @MUTED@; }
.np-swap .arrow { color: @BRAND@; font-weight: 700; }
.np-timeline { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.np-step { border-top: 4px solid @BRAND@; background: @SURFACE@; border-radius: 0 0 12px 12px; border: 1px solid @LINE@; border-top: 4px solid @BRAND@; padding: 0.9rem 1.2rem 1.1rem; }
.np-step span { font-family: 'Young Serif', Georgia, serif; color: @BRAND@; font-size: 1.1rem; display: block; margin-bottom: 0.3rem; }
.np-two { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
.np-footer { color: @MUTED@; font-size: 0.85rem; line-height: 1.5; border-top: 1px solid @LINE@; padding-top: 1rem; margin-top: 2.2rem; max-width: 52rem; }
@media (max-width: 760px) {
.np-wide { display: none; }
.np-narrow { display: block; }
.np-grid3, .np-timeline, .np-two { grid-template-columns: 1fr; }
.np-meal { grid-template-columns: 1fr; gap: 0.2rem; }
.np-wordmark { font-size: 2rem; }
}
"""
for _k, _v in {"INK": INK, "MUTED": MUTED, "LINE": LINE, "BG": BG, "SURFACE": SURFACE,
               "BRAND": BRAND, "BRAND_DARK": BRAND_DARK, "BRAND_TINT": BRAND_TINT,
               "FONT": FONT_STACK}.items():
    THEME_CSS = THEME_CSS.replace(f"@{_k}@", _v)
st.markdown("<style>" + "\n".join(l.strip() for l in THEME_CSS.splitlines() if l.strip())
            + "</style>", unsafe_allow_html=True)


def plate_logo(size=40):
    """A small plate: half vegetables, a quarter protein, a quarter grains (the plate method)."""
    r, c = 18, 20
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 40 40" role="img" aria-label="NutriPath logo">'
            f'<path d="M{c-r},{c} A{r},{r} 0 0 1 {c+r},{c} Z" fill="{BRAND}"/>'
            f'<path d="M{c},{c} L{c-r},{c} A{r},{r} 0 0 0 {c},{c+r} Z" fill="#7DB9A1"/>'
            f'<path d="M{c},{c} L{c},{c+r} A{r},{r} 0 0 0 {c+r},{c} Z" fill="#CFE5DA"/>'
            f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{INK}" stroke-width="2"/></svg>')


def _ruler_svg(now, predicted, W, x0, x1, label_fs, zone_fs, cls, uid):
    lo, hi = 5.0, 12.0
    X = lambda v: x0 + (min(max(v, lo), hi) - lo) / (hi - lo) * (x1 - x0)
    zones = [("Low", 5, 7), ("Moderate", 7, 8.5), ("High", 8.5, 10), ("Critical", 10, 12)]
    s = [f'<svg class="np-ruler {cls}" viewBox="0 0 {W} 124" role="img" '
         f'aria-label="A1C scale. Now {now:.1f} percent. Predicted in 3 months {predicted:.1f} percent.">',
         f'<defs><clipPath id="{uid}"><rect x="{x0}" y="46" width="{x1 - x0}" height="26" rx="13"/></clipPath></defs>',
         f'<g clip-path="url(#{uid})">']
    for name, a, b in zones:
        s.append(f'<rect x="{X(a):.1f}" y="46" width="{X(b) - X(a):.1f}" height="26" fill="{ZONE_FILL[name]}"/>')
    for a in (7, 8.5, 10):
        s.append(f'<rect x="{X(a) - 1:.1f}" y="46" width="2" height="26" fill="#fff"/>')
    s.append("</g>")
    for name, a, b in zones:
        s.append(f'<text x="{(X(a) + X(b)) / 2:.1f}" y="63.5" text-anchor="middle" font-size="{zone_fs}" '
                 f'fill="{INK}" font-family="{FONT_STACK}">{name}</text>')
    xn, xp = X(now), X(predicted)
    lab = lambda x: min(max(x, 72), W - 72)
    s.append(f'<path d="M{xn - 7:.1f},26 L{xn + 7:.1f},26 L{xn:.1f},41 Z" fill="{INK}"/>')
    s.append(f'<text x="{lab(xn):.1f}" y="18" text-anchor="middle" font-size="{label_fs}" font-weight="700" '
             f'fill="{INK}" font-family="{FONT_STACK}">Now {now:.1f}%</text>')
    s.append(f'<path d="M{xp - 7:.1f},92 L{xp + 7:.1f},92 L{xp:.1f},77 Z" fill="{BRAND}"/>')
    s.append(f'<text x="{lab(xp):.1f}" y="112" text-anchor="middle" font-size="{label_fs}" font-weight="700" '
             f'fill="{BRAND}" font-family="{FONT_STACK}">In 3 months {predicted:.1f}%</text>')
    s.append("</svg>")
    return "".join(s)


def a1c_ruler(now, predicted):
    """The one memorable visual: where you are on the A1C scale now, and in 3 months.
    A wide version for desktop and a larger-text narrow version for phones."""
    return (_ruler_svg(now, predicted, 700, 30, 670, 14.5, 12.5, "np-wide", "npTrackWide")
            + _ruler_svg(now, predicted, 360, 16, 344, 14, 11.5, "np-narrow", "npTrackNarrow"))


def style_fig(fig, height=340, legend=False):
    """Quiet, readable charts: no chart junk, light grid, same font as the page."""
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=20, b=8), title=None,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT_STACK, color=INK, size=13), showlegend=legend,
                      legend=dict(orientation="h", y=1.14, x=0, title=None),
                      hoverlabel=dict(font_family=FONT_STACK, bgcolor="#fff"))
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickcolor=LINE)
    fig.update_yaxes(gridcolor="#E6ECE8", zeroline=False, linecolor="rgba(0,0,0,0)")
    return fig

# ── NutriPath core logic (identical in the notebook and in app.py) ──────────
import re
import numpy as np
import pandas as pd

# One model name for the whole project. If it is ever unavailable, the
# generator below falls back to the next model, then to a rules-based plan.
PLAN_MODEL = "qwen/qwen3.8-27b"
FALLBACK_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

# The 41 features, in the exact order the model was trained on (NB2)
FEATURE_COLUMNS = [
    "age", "bmi", "a1c_percent", "diabetes_duration_yrs",
    "fasting_glucose_mgdl", "systolic_bp", "diastolic_bp",
    "egfr", "ldl_mgdl", "hdl_mgdl", "triglycerides_mgdl",
    "sleep_hours_per_night", "caloric_intake_kcal",
    "carb_intake_grams", "meals_per_day",
    "fruit_veg_servings_per_day", "dietary_adherence_score_1to10",
    "activity_level_encoded", "glycemic_control_encoded",
    "sleep_quality_encoded", "motivation_encoded",
    "sex_encoded", "gi_awareness_encoded", "ckd_stage_encoded",
    "on_metformin", "on_glp1_agonist", "on_insulin",
    "on_sglt2_inhibitor", "on_statin",
    "hypertension", "dyslipidemia", "obesity",
    "peripheral_neuropathy", "retinopathy",
    "motivation_score", "dietary_pattern_score",
    "medication_count", "comorbidity_count",
    "high_risk_flag", "carb_load_ratio", "sleep_deprived",
]

# Encoding maps (must match the Gold layer in NB1)
ACTIVITY_MAP = {"Sedentary": 0, "Lightly active": 1,
                "Moderately active": 2, "Very active": 3}
GLYCEMIC_MAP = {"controlled": 0, "moderate": 1, "poor": 2}
SLEEP_QUALITY_MAP = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
MOTIVATION_MAP = {"Low": 0, "Moderate": 1, "High": 2}
GI_MAP = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}
CKD_MAP = {"G1 (normal)": 0, "G2 (mild)": 1, "G3a (mild-mod)": 2,
           "G3b (mod-severe)": 3, "G4 (severe)": 4, "G5 (kidney failure)": 5}
DIET_SCORE_MAP = {"Mediterranean": 0.90, "Plant-based": 0.85, "Low-carb": 0.80,
                  "Intermittent fasting": 0.75, "Low-fat": 0.70,
                  "No specific pattern": 0.60, "Standard American": 0.50}
MOTIVATION_SCORE_MAP = {"Low": 0.85, "Moderate": 1.0, "High": 1.25}


def encode_patient(p):
    """Turn one patient dict (plain-English form values) into the 41-feature row."""
    a1c = p["a1c"]
    glycemic = "controlled" if a1c < 7.0 else "moderate" if a1c < 10.0 else "poor"

    meds = [p["on_metformin"], p["on_glp1"], p["on_insulin"], p["on_sglt2"], p["on_statin"]]
    conds = [p["hypertension"], p["dyslipidemia"], p["obesity"],
             p["neuropathy"], p["retinopathy"]]

    row = {
        "age": p["age"],
        "bmi": p["bmi"],
        "a1c_percent": a1c,
        "diabetes_duration_yrs": p["diabetes_duration"],
        "fasting_glucose_mgdl": p["fasting_glucose"],
        "systolic_bp": p["systolic_bp"],
        "diastolic_bp": p["diastolic_bp"],
        "egfr": p["egfr"],
        "ldl_mgdl": p["ldl"],
        "hdl_mgdl": p["hdl"],
        "triglycerides_mgdl": p["triglycerides"],
        "sleep_hours_per_night": p["sleep_hours"],
        "caloric_intake_kcal": p["caloric_intake"],
        "carb_intake_grams": p["carb_intake"],
        "meals_per_day": p["meals_per_day"],
        "fruit_veg_servings_per_day": p["fruit_veg"],
        "dietary_adherence_score_1to10": p["adherence"],
        "activity_level_encoded": ACTIVITY_MAP[p["activity"]],
        "glycemic_control_encoded": GLYCEMIC_MAP[glycemic],
        "sleep_quality_encoded": SLEEP_QUALITY_MAP[p["sleep_quality"]],
        "motivation_encoded": MOTIVATION_MAP[p["motivation"]],
        "sex_encoded": 1 if p["sex"] == "Male" else 0,
        "gi_awareness_encoded": GI_MAP[p["gi_awareness"]],
        "ckd_stage_encoded": CKD_MAP[p["ckd_stage"]],
        "on_metformin": int(p["on_metformin"]),
        "on_glp1_agonist": int(p["on_glp1"]),
        "on_insulin": int(p["on_insulin"]),
        "on_sglt2_inhibitor": int(p["on_sglt2"]),
        "on_statin": int(p["on_statin"]),
        "hypertension": int(p["hypertension"]),
        "dyslipidemia": int(p["dyslipidemia"]),
        "obesity": int(p["obesity"]),
        "peripheral_neuropathy": int(p["neuropathy"]),
        "retinopathy": int(p["retinopathy"]),
        "motivation_score": MOTIVATION_SCORE_MAP[p["motivation"]],
        "dietary_pattern_score": DIET_SCORE_MAP[p["dietary_pattern"]],
        "medication_count": int(sum(meds)),
        "comorbidity_count": int(sum(conds)),
        "high_risk_flag": int(a1c > 9.0 and ACTIVITY_MAP[p["activity"]] == 0
                              and p["adherence"] < 4),
        "carb_load_ratio": (p["carb_intake"] * 4) / p["caloric_intake"],
        "sleep_deprived": int(p["sleep_hours"] < 6),
    }
    # Column order matters for XGBoost, so enforce it here
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def predict_reduction(model, features):
    """Predicted A1C drop (percentage points) over 3 months. Never negative."""
    reduction = float(model.predict(features[FEATURE_COLUMNS])[0])
    return round(max(0.0, reduction), 3)


def get_risk_tier(predicted_a1c):
    """Risk label and emoji from the predicted A1C at 3 months."""
    if predicted_a1c < 7.0:
        return "Low", "🟢"
    elif predicted_a1c < 8.5:
        return "Moderate", "🟡"
    elif predicted_a1c < 10.0:
        return "High", "🟠"
    return "Critical", "🔴"

# ── Layer 1: rules-based findings and safety notes (no LLM involved) ────────
def describe_meds_conditions(p):
    """Plain-English lists of the patient's medicines and conditions."""
    meds = [name for key, name in [
        ("on_metformin", "metformin"), ("on_glp1", "a GLP-1 medicine"),
        ("on_insulin", "insulin"), ("on_sglt2", "an SGLT2 medicine"),
        ("on_statin", "a statin")] if p[key]]
    conds = [name for key, name in [
        ("hypertension", "high blood pressure"), ("dyslipidemia", "high cholesterol"),
        ("obesity", "obesity"), ("neuropathy", "nerve damage in the feet or hands"),
        ("retinopathy", "eye damage from diabetes")] if p[key]]
    return (", ".join(meds) or "none listed"), (", ".join(conds) or "none listed")


def build_findings(p):
    """Return (findings, safety). Findings drive the plan; safety is added by code."""
    findings, safety = [], []

    # Blood sugar
    gap = p["a1c"] - 7.0
    if gap > 0:
        findings.append(f"A1C is {p['a1c']}%, which is {gap:.1f} points above the 7% goal "
                        f"many adults aim for.")
    else:
        findings.append(f"A1C is {p['a1c']}%, which is at or under the 7% goal. "
                        f"The aim is to keep it there.")

    # Food
    carb_pct = p["carb_intake"] * 4 / p["caloric_intake"] * 100
    if carb_pct > 45:
        findings.append(f"Carbs make up about {carb_pct:.0f}% of daily calories "
                        f"({p['carb_intake']} g of {p['caloric_intake']} kcal). "
                        f"Cutting toward 40-45% usually helps blood sugar.")
    if p["fruit_veg"] < 5:
        findings.append(f"Eats about {p['fruit_veg']} servings of fruit and vegetables a day. "
                        f"The goal is 5 or more.")
    if p["dietary_pattern"] in ("Standard American", "No specific pattern"):
        findings.append(f"Current eating style is '{p['dietary_pattern']}', "
                        f"so there is room to build a steadier routine.")
    if p["gi_awareness"] in ("None", "Low"):
        findings.append("Knows little about glycemic index (how fast foods raise blood sugar).")
    if p["adherence"] < 6:
        findings.append(f"Sticks to the current diet {p['adherence']} out of 10. "
                        f"Plans need to be small and easy to keep.")

    # Lifestyle
    if p["activity"] in ("Sedentary", "Lightly active"):
        findings.append(f"Activity level is '{p['activity']}'. Adding short walks is the easiest win.")
    if p["sleep_hours"] < 7:
        findings.append(f"Sleeps about {p['sleep_hours']} hours a night. "
                        f"Short sleep can push blood sugar up.")

    # Heart numbers
    if p["systolic_bp"] >= 130 or p["diastolic_bp"] >= 80:
        findings.append(f"Blood pressure is {p['systolic_bp']}/{p['diastolic_bp']}, "
                        f"above the 130/80 goal. Less salt and more potassium-rich foods help.")
    if p["ldl"] >= 100:
        findings.append(f"LDL (bad) cholesterol is {p['ldl']} mg/dL, above the 100 goal.")
    if p["triglycerides"] >= 150:
        findings.append(f"Triglycerides are {p['triglycerides']} mg/dL, above the 150 goal. "
                        f"Sweet drinks, refined carbs and alcohol raise them.")
    hdl_goal = 40 if p["sex"] == "Male" else 50
    if p["hdl"] < hdl_goal:
        findings.append(f"HDL (good) cholesterol is {p['hdl']} mg/dL, under the {hdl_goal} goal.")

    # Safety notes (added to the final plan by code, not written by the LLM)
    if p["on_insulin"]:
        safety.append("You take insulin. Skipping meals or adding more exercise can cause low "
                      "blood sugar. Ask your care team how to adjust, and keep a fast sugar "
                      "source (juice or glucose tabs) with you.")
    if p["on_glp1"]:
        safety.append("You take a GLP-1 medicine. It lowers appetite, so eat small, protein-rich "
                      "meals and drink water. Tell your doctor if you feel sick or can't eat.")
    if CKD_MAP[p["ckd_stage"]] >= 2:
        safety.append(f"Your kidney stage is {p['ckd_stage']}. Ask your doctor how much protein, "
                      f"salt and potassium is right for you before changing your diet.")
    if p["neuropathy"]:
        safety.append("You have nerve damage. Check your feet daily, wear supportive shoes, and "
                      "choose low-impact exercise such as walking, cycling or swimming.")
    if p["retinopathy"]:
        safety.append("You have eye damage from diabetes. Avoid heavy lifting and hard straining, "
                      "and keep up your yearly eye exams.")
    return findings, safety

# ── Layer 2: the LLM only writes plain English from the findings above ──────
# The LLM returns structured JSON (not a wall of text) so the app can lay the
# plan out as cards. If the JSON is broken, we try the next model, then a
# rules-based plan with the same structure.
import json

SYSTEM_PROMPT = """You are NutriPath, a friendly diabetes lifestyle coach.
You write for a regular person with no medical training.

Rules:
- Write at a 6th-grade reading level. Keep sentences under 15 words.
- Use everyday words. If you must use a medical word, explain it in a few words.
- Be specific. Name real foods and real portions (for example "1/2 cup of beans"),
  never vague advice like "eat healthy" or "reduce carbs".
- No filler, no greetings, no long introductions, no motivational speeches.
- Respect every food restriction and the patient's eating style.
- Never tell the patient to start, stop or change a medicine.
- Use only the facts you are given. Do not invent numbers or lab values.
- Reply with ONE valid JSON object and nothing else. No markdown, no code fences."""

SCHEMA_TEXT = """Return JSON in exactly this shape:
{
  "snapshot": "2 short sentences: where they are now and where they could be in 3 months.",
  "priorities": [
    {"action": "One clear action with a real food or portion.", "why": "Why this fits THIS patient, tied to a key finding."},
    {"action": "...", "why": "..."},
    {"action": "...", "why": "..."}
  ],
  "eating": {
    "breakfast": "Real foods and portions.",
    "lunch": "Real foods and portions.",
    "dinner": "Real foods and portions.",
    "snack": "Real foods and portions.",
    "swaps": [
      {"from": "food they likely eat now", "to": "better choice with a portion"},
      {"from": "...", "to": "..."},
      {"from": "...", "to": "..."}
    ]
  },
  "movement": [
    {"when": "Weeks 1-2", "plan": "Activity, minutes, days per week."},
    {"when": "Weeks 3-4", "plan": "..."},
    {"when": "Weeks 5-12", "plan": "..."}
  ],
  "sleep": ["Short practical tip.", "Short practical tip."],
  "doctor_questions": ["Short question.", "Short question.", "Short question."]
}
Exactly 3 priorities, exactly 3 swaps, exactly 3 movement steps, 3 doctor questions."""


def build_prompt(p, predicted_reduction, findings, safety):
    meds, conds = describe_meds_conditions(p)
    predicted_a1c = max(p["a1c"] - predicted_reduction, 0)
    findings_text = "\n".join(f"- {f}" for f in findings)
    safety_text = "\n".join(f"- {s}" for s in safety) or "- none"
    facts = f"""PATIENT FACTS
- Age {p['age']}, {p['sex']}, BMI {p['bmi']}, diabetes for {p['diabetes_duration']} years
- Current A1C {p['a1c']}%. Model-predicted A1C in 3 months: {predicted_a1c:.1f}%
- Eating style: {p['dietary_pattern']}. Food restrictions: {p['restrictions']}
- Medicines: {meds}. Conditions: {conds}

KEY FINDINGS (already checked by rules, use these)
{findings_text}

SAFETY CONTEXT (do not repeat, but never contradict)
{safety_text}

"""
    return facts + SCHEMA_TEXT


def _clean_llm_text(text):
    """Remove <think> blocks that some models add. Unclosed <think> counts as empty."""
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def _txt(x):
    return str(x).strip() if x is not None else ""


def _validate_plan(d):
    """Check the LLM's JSON has everything the app needs. Raises ValueError if not."""
    if not isinstance(d, dict):
        raise ValueError("plan is not an object")
    snapshot = _txt(d.get("snapshot"))
    priorities = [{"action": _txt(x.get("action")), "why": _txt(x.get("why"))}
                  for x in (d.get("priorities") or []) if isinstance(x, dict) and _txt(x.get("action"))][:3]
    eating_in = d.get("eating") if isinstance(d.get("eating"), dict) else {}
    meals = {k: _txt(eating_in.get(k)) for k in ("breakfast", "lunch", "dinner", "snack")}
    swaps = [{"from": _txt(x.get("from")), "to": _txt(x.get("to"))}
             for x in (eating_in.get("swaps") or [])
             if isinstance(x, dict) and _txt(x.get("from")) and _txt(x.get("to"))][:3]
    movement = [{"when": _txt(x.get("when")), "plan": _txt(x.get("plan"))}
                for x in (d.get("movement") or [])
                if isinstance(x, dict) and _txt(x.get("when")) and _txt(x.get("plan"))][:3]
    sleep = [_txt(s) for s in (d.get("sleep") or []) if _txt(s)][:3]
    questions = [_txt(q) for q in (d.get("doctor_questions") or []) if _txt(q)][:4]

    if not snapshot or len(priorities) < 3 or not all(meals.values()) or len(movement) < 3:
        raise ValueError("plan is missing required sections")
    if not sleep:
        sleep = ["Go to bed and wake up at the same time each day.",
                 "Turn off screens 30 minutes before bed."]
    if not questions:
        questions = ["Is my A1C goal right for me?",
                     "Should my medicines change as my habits change?",
                     "How often should I get my A1C checked?"]
    return {"snapshot": snapshot, "priorities": priorities,
            "eating": {**meals, "swaps": swaps},
            "movement": movement, "sleep": sleep, "doctor_questions": questions}


def _parse_plan_json(text):
    text = _clean_llm_text(text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON found in the reply")
    return _validate_plan(json.loads(text[start:end + 1]))


def build_fallback_plan(p, predicted_reduction, findings):
    """Rules-based plan with the same structure, used only if every LLM call fails."""
    predicted_a1c = max(p["a1c"] - predicted_reduction, 0)
    carb_pct = p["carb_intake"] * 4 / p["caloric_intake"] * 100

    candidates = []
    if carb_pct > 45:
        candidates.append(("Cut starchy sides (rice, pasta, bread) by about one third at each meal.",
                           f"Carbs are about {carb_pct:.0f}% of your calories."))
    if p["fruit_veg"] < 5:
        candidates.append(("Add 1 extra cup of vegetables to lunch and dinner.",
                           f"You eat about {p['fruit_veg']} servings a day. The goal is 5 or more."))
    if p["activity"] in ("Sedentary", "Lightly active"):
        candidates.append(("Walk 10 minutes after your biggest meal.",
                           "Short walks help your body use sugar."))
    if p["sleep_hours"] < 7:
        candidates.append(("Go to bed 30 minutes earlier this week.",
                           f"You sleep about {p['sleep_hours']} hours. Short sleep raises blood sugar."))
    if p["triglycerides"] >= 150:
        candidates.append(("Swap sweet drinks for water or unsweetened tea.",
                           "Sweet drinks raise blood sugar and triglycerides."))
    if p["systolic_bp"] >= 130 or p["diastolic_bp"] >= 80:
        candidates.append(("Cut back on salty packaged foods and take-out.",
                           "Your blood pressure is above the 130/80 goal."))
    candidates += [
        ("Fill half your plate with non-starchy vegetables.", "Vegetables add fiber without raising blood sugar much."),
        ("Walk 10 minutes after your biggest meal.", "Short walks help your body use sugar."),
        ("Drink water instead of sweet drinks.", "Sweet drinks spike blood sugar fast."),
    ]
    priorities, seen = [], set()
    for action, why in candidates:
        if action not in seen:
            seen.add(action)
            priorities.append({"action": action, "why": why})
        if len(priorities) == 3:
            break

    if p["activity"] in ("Sedentary", "Lightly active"):
        movement = [{"when": "Weeks 1-2", "plan": "Walk 10 minutes, 3 days a week."},
                    {"when": "Weeks 3-4", "plan": "Walk 15 minutes, 4 days a week."},
                    {"when": "Weeks 5-12", "plan": "Walk 20 to 30 minutes, 5 days a week."}]
    else:
        movement = [{"when": "Weeks 1-2", "plan": "Keep your usual routine. Aim for 150 minutes a week."},
                    {"when": "Weeks 3-4", "plan": "Add 2 short strength sessions a week."},
                    {"when": "Weeks 5-12", "plan": "Keep both going and add 5 minutes to your longest session."}]

    return {
        "snapshot": (f"Your A1C is {p['a1c']}%. With steady habits, it could reach about "
                     f"{predicted_a1c:.1f}% in 3 months."),
        "priorities": priorities,
        "eating": {
            "note": ("Our AI coach was unavailable, so here is the simple plate method. "
                     "Fill half your plate with non-starchy vegetables. Fill one quarter with "
                     "protein (fish, chicken, eggs, tofu or beans). Fill one quarter with whole "
                     "grains or starchy vegetables. Adjust for your food restrictions."),
            "swaps": [{"from": "Soda or sweet tea", "to": "Water or unsweetened tea"},
                      {"from": "White rice", "to": "1/2 cup brown rice"},
                      {"from": "Chips", "to": "Carrot sticks with hummus"}],
        },
        "movement": movement,
        "sleep": ["Go to bed and wake up at the same time each day.",
                  "Turn off screens 30 minutes before bed."],
        "doctor_questions": ["Is my A1C goal right for me?",
                             "Should my medicines change as my habits change?",
                             "How often should I get my A1C checked?"],
    }


def generate_plan(client, p, predicted_reduction, findings, safety):
    """Try the main model, then fallbacks, then the rules-based plan.
    Returns (plan_dict, model_used)."""
    prompt = build_prompt(p, predicted_reduction, findings, safety)
    for model_name in [PLAN_MODEL] + FALLBACK_MODELS:
        for use_json_mode in (True, False):
            try:
                kwargs = dict(
                    model=model_name,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    max_tokens=1800,
                    temperature=0.4,
                )
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                return _parse_plan_json(resp.choices[0].message.content), model_name
            except ValueError:
                break      # the model answered but the JSON was unusable: try the next model
            except Exception:
                continue   # API problem (for example, JSON mode not supported): retry or move on
    return build_fallback_plan(p, predicted_reduction, findings), "rules-based fallback"

# ── Pipeline: form values in, prediction + findings + plan out ──────────────
DISCLAIMER = ("Please share this plan with your doctor or diabetes educator before "
              "changing your diet, activity, or medicines.")


def plan_to_markdown(plan, safety, a1c_now, predicted_a1c, reduction, tier, emoji):
    """Plain-text version of the plan, used for the download button and the notebook tests."""
    lines = ["# Your NutriPath plan", "",
             "## Your snapshot",
             f"- A1C now: {a1c_now}%",
             f"- Predicted A1C in 3 months: {predicted_a1c:.1f}% (drop of {reduction:.2f} points)",
             f"- Risk tier: {emoji} {tier}", "", plan["snapshot"], ""]
    if safety:
        lines += ["## Safety notes for you"] + [f"- {s}" for s in safety] + [""]
    lines += ["## 3 priorities this week"]
    for i, item in enumerate(plan["priorities"], 1):
        lines.append(f"{i}. {item['action']}")
        if item["why"]:
            lines.append(f"   Why for you: {item['why']}")
    lines += ["", "## A sample day of eating"]
    eating = plan["eating"]
    if eating.get("breakfast"):
        lines += [f"- Breakfast: {eating['breakfast']}", f"- Lunch: {eating['lunch']}",
                  f"- Dinner: {eating['dinner']}", f"- Snack: {eating['snack']}"]
    elif eating.get("note"):
        lines.append(eating["note"])
    if eating.get("swaps"):
        lines += ["", "Easy swaps:"] + [f"- {s['from']} -> {s['to']}" for s in eating["swaps"]]
    lines += ["", "## Your movement plan"] + [f"- {m['when']}: {m['plan']}" for m in plan["movement"]]
    lines += ["", "## Sleep"] + [f"- {s}" for s in plan["sleep"]]
    lines += ["", "## Questions to ask your doctor"] + [f"- {q}" for q in plan["doctor_questions"]]
    lines += ["", f"_{DISCLAIMER}_"]
    return "\n".join(lines)


def run_pipeline(model, client, p):
    features = encode_patient(p)
    reduction = predict_reduction(model, features)
    predicted_a1c = round(max(p["a1c"] - reduction, 0.0), 2)
    tier, emoji = get_risk_tier(predicted_a1c)

    findings, safety = build_findings(p)
    plan, model_used = generate_plan(client, p, reduction, findings, safety)

    return {
        "a1c_now": p["a1c"],
        "reduction": reduction,
        "predicted_a1c": predicted_a1c,
        "risk_tier": tier,
        "risk_emoji": emoji,
        "plan": plan,                      # structured dict, shown as cards in the app
        "plan_markdown": plan_to_markdown(plan, safety, p["a1c"], predicted_a1c,
                                          reduction, tier, emoji),
        "findings": findings,
        "safety": safety,                  # always shown by code, never written by the LLM
        "model_used": model_used,
    }


# ══════════════════════════ Streamlit UI ═════════════════════════════════════
@st.cache_resource
def load_model():
    m = XGBRegressor()
    m.load_model(str(BASE_DIR / "model" / "nutripath_xgb.json"))
    return m


@st.cache_data
def load_dashboard_data():
    return pd.read_parquet(BASE_DIR / "data" / "dashboard_data.parquet")


model = load_model()
df = load_dashboard_data()
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    client = None  # the app still works: plans fall back to the rules-based version

TIER_COLORS = RISK
GLYCEMIC_COLORS = {"Controlled": RISK["Low"], "Moderate": RISK["High"], "Poor": RISK["Critical"]}
ACTIVITY_ORDER = ["Sedentary", "Lightly active", "Moderately active", "Very active"]
MOTIVATION_ORDER = ["Low", "Moderate", "High"]

st.markdown(
    f'<div class="np-hero"><div class="np-brand">{plate_logo()}<span class="np-wordmark">NutriPath</span></div>'
    '<p class="np-tag">Enter your health numbers. Get a plan you can read in two minutes.</p></div>',
    unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Population dashboard", "My personal plan"])


# ── Tab 1: dashboard ────────────────────────────────────────────────────────
def avg_bar(data, group_col, order, horizontal=False):
    summary = (data.groupby(group_col)
               .agg(avg_reduction=("predicted_a1c_reduction_3mo", "mean"),
                    patients=("predicted_a1c_reduction_3mo", "count"))
               .reset_index())
    if order:
        summary[group_col] = summary[group_col].astype(
            "category").cat.set_categories(order, ordered=True)
        summary = summary.sort_values(group_col)
    else:
        summary = summary.sort_values("avg_reduction", ascending=horizontal)
    x, y = ("avg_reduction", group_col) if horizontal else (group_col, "avg_reduction")
    fig = px.bar(summary, x=x, y=y, custom_data=["patients"],
                 orientation="h" if horizontal else "v",
                 labels={"avg_reduction": "Average predicted A1C drop (points)", group_col: ""})
    fig.update_traces(marker_color=BAR, texttemplate="%{x:.2f}" if horizontal else "%{y:.2f}",
                      textposition="outside", cliponaxis=False,
                      hovertemplate="<b>%{y}</b><br>Average drop: %{x:.2f} points<br>%{customdata[0]:,} profiles<extra></extra>"
                      if horizontal else
                      "<b>%{x}</b><br>Average drop: %{y:.2f} points<br>%{customdata[0]:,} profiles<extra></extra>")
    fig = style_fig(fig, height=330)
    if horizontal:
        fig.update_xaxes(range=[0, summary["avg_reduction"].max() * 1.2])
    return fig


with tab1:
    st.markdown("### Who are our patients, and what helps most?")
    c1, c2, c3 = st.columns(3)
    glyc = c1.multiselect("Blood sugar group", list(GLYCEMIC_COLORS),
                          default=list(GLYCEMIC_COLORS))
    act = c2.multiselect("Activity level", ACTIVITY_ORDER, default=ACTIVITY_ORDER)
    mot = c3.multiselect("Motivation", MOTIVATION_ORDER, default=MOTIVATION_ORDER)

    view = df[df["glycemic_label"].isin(glyc) & df["activity_label"].isin(act)
              & df["motivation_label"].isin(mot)]

    if view.empty:
        st.warning("No patients match these filters. Try selecting more options.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Patients", f"{len(view):,}")
        k2.metric("Average A1C", f"{view['a1c_percent'].mean():.2f}%")
        k3.metric("Avg predicted drop", f"{view['predicted_a1c_reduction_3mo'].mean():.3f} pts")
        k4.metric("Not at goal (A1C 7%+)",
                  f"{(view['a1c_percent'] >= 7.0).mean() * 100:.1f}%")

        st.markdown("#### A1C distribution by blood sugar group")
        fig1 = px.histogram(view, x="a1c_percent", color="glycemic_label", nbins=40,
                            color_discrete_map=GLYCEMIC_COLORS, barmode="overlay", opacity=0.8,
                            category_orders={"glycemic_label": list(GLYCEMIC_COLORS)},
                            labels={"a1c_percent": "A1C (%)", "glycemic_label": "Group",
                                    "count": "Patients"})
        fig1.add_vline(x=7.0, line_dash="dash", line_color=INK,
                       annotation_text="7% goal", annotation_font_color=INK)
        fig1.update_yaxes(title="Patients")
        st.plotly_chart(style_fig(fig1, height=320, legend=True))

        left, right = st.columns(2)
        with left:
            st.markdown("#### Predicted drop by eating style")
            st.plotly_chart(avg_bar(view, "diet_label", None, horizontal=True))
            st.markdown("#### Predicted drop by motivation")
            st.plotly_chart(avg_bar(view, "motivation_label", MOTIVATION_ORDER))
        with right:
            st.markdown("#### Predicted drop by activity level")
            st.plotly_chart(avg_bar(view, "activity_label", ACTIVITY_ORDER))
            st.markdown("#### Predicted risk after 3 months")
            tiers = view["risk_tier"].value_counts().reindex(
                list(TIER_COLORS)).fillna(0).reset_index()
            tiers.columns = ["risk_tier", "patients"]
            fig5 = px.bar(tiers, x="risk_tier", y="patients", color="risk_tier",
                          color_discrete_map=TIER_COLORS,
                          labels={"risk_tier": "", "patients": "Patients"})
            fig5.update_traces(texttemplate="%{y:,}", textposition="outside", cliponaxis=False,
                               hovertemplate="<b>%{x}</b><br>%{y:,} patients<extra></extra>")
            st.plotly_chart(style_fig(fig5, height=330))
        st.caption("These charts show how the NutriPath model behaves across 100,000 "
                   "example profiles.")


def render_result(result):
    plan = result["plan"]
    tier = result["risk_tier"]

    # Safety notes first: they matter most and are written by code, not the LLM
    if result["safety"]:
        items = "".join(f"<li>{h(s)}</li>" for s in result["safety"])
        st.markdown(f'<div class="np-safety"><b>Before you start, please note</b><ul>{items}</ul></div>',
                    unsafe_allow_html=True)

    st.markdown(
        '<div class="np-panel"><h3 class="np-ptitle">Your 3-month outlook</h3>'
        + a1c_ruler(result["a1c_now"], result["predicted_a1c"])
        + '<div class="np-stats">'
        f'<div><span>Predicted drop</span><b>{result["reduction"]:.2f} points</b></div>'
        f'<div><span>Predicted A1C</span><b>{result["predicted_a1c"]:.1f}%</b></div>'
        f'<div><span>Risk tier</span><b><i class="np-dot" style="background:{RISK[tier]}"></i>{tier}</b></div>'
        '</div>'
        f'<p class="np-snapshot">{h(plan["snapshot"])}</p></div>',
        unsafe_allow_html=True)

    st.markdown("### Your 3 priorities this week")
    cards = "".join(
        f'<div class="np-card"><div class="np-num">{i}</div><div class="np-action">{h(item["action"])}</div>'
        + (f'<div class="np-why">Why for you: {h(item["why"])}</div>' if item["why"] else "")
        + "</div>" for i, item in enumerate(plan["priorities"], 1))
    st.markdown(f'<div class="np-grid3">{cards}</div>', unsafe_allow_html=True)

    st.markdown("### A sample day of eating")
    eating = plan["eating"]
    if eating.get("breakfast"):
        rows = "".join(f'<div class="np-meal"><span>{label}</span><div>{h(eating[key])}</div></div>'
                       for label, key in [("Breakfast", "breakfast"), ("Lunch", "lunch"),
                                          ("Dinner", "dinner"), ("Snack", "snack")])
        st.markdown(f'<div class="np-menu">{rows}</div>', unsafe_allow_html=True)
    elif eating.get("note"):
        st.markdown(f'<div class="np-note">{h(eating["note"])}</div>', unsafe_allow_html=True)
    if eating.get("swaps"):
        rows = "".join(f'<div class="np-swap"><span class="from">{h(s["from"])}</span>'
                       f'<span class="arrow">→</span><b>{h(s["to"])}</b></div>' for s in eating["swaps"])
        st.markdown(f'<div class="np-swaps"><h4>Easy swaps</h4>{rows}</div>', unsafe_allow_html=True)

    st.markdown("### Your movement plan")
    steps = "".join(f'<div class="np-step"><span>{h(m["when"])}</span>{h(m["plan"])}</div>'
                    for m in plan["movement"])
    st.markdown(f'<div class="np-timeline">{steps}</div>', unsafe_allow_html=True)

    sleep = "".join(f"<li>{h(t)}</li>" for t in plan["sleep"])
    qs = "".join(f"<li>{h(q)}</li>" for q in plan["doctor_questions"])
    st.markdown(f'<div class="np-two"><div><h3>Sleep</h3><ul class="np-list">{sleep}</ul></div>'
                f'<div><h3>Questions to ask your doctor</h3><ul class="np-list">{qs}</ul></div></div>',
                unsafe_allow_html=True)

    st.markdown(f'<div class="np-note">{h(DISCLAIMER)}</div>', unsafe_allow_html=True)

    with st.expander("Why this plan? (what we found in your numbers)"):
        st.markdown("\n".join(f"- {str(f).replace('$', chr(92) + '$')}" for f in result["findings"]))
    st.download_button("Download my plan", data=result["plan_markdown"],
                       file_name="my_nutripath_plan.md", mime="text/markdown")


# ── Tab 2: personal plan ────────────────────────────────────────────────────
RESTRICTION_OPTIONS = ["Vegetarian", "Vegan", "Gluten-free", "Dairy-free",
                       "Nut allergy", "Halal", "Kosher", "Low-sodium"]

with tab2:
    st.subheader("Tell us about you")

    with st.expander("How to use this form (read this first)", expanded=True):
        st.markdown(
            """
**3 steps**
1. **Replace every example value.** The form starts with a sample patient. Change each
   number and choice to match you.
2. **Hover over the ⓘ icon** next to a field to see where to find that number.
3. **Click "Get my plan".** You will get a predicted A1C change, a risk tier and a plain-English plan.

**Where to find your numbers**

| What you need | Where to look |
|---|---|
| A1C, fasting glucose, kidney (eGFR), LDL, HDL, triglycerides | Your latest **lab report** or patient portal (MyChart or similar). Ask your doctor's office to read them out if you can't find them. |
| Blood pressure | A home cuff, a pharmacy kiosk, or your last **visit summary**. |
| BMI | Your visit summary, or use a free BMI calculator with your height and weight. |
| Medicines | Your **pill bottles or pharmacy app**. The brand and generic names are on the label. |
| Conditions (kidney stage, nerve or eye damage) | Your doctor's diagnosis list in the patient portal. If you are not sure, leave it unchecked and ask your doctor. |
| Calories, carbs, servings | Rough estimates are fine. Use a food-tracking app or the nutrition labels on packages. |

*Don't have a recent lab result? Use your most recent one and note the date. Labs older than
a year may give a less useful result.*
"""
        )

    with st.form("patient_form"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**About you**")
            age = st.number_input("Age", 18, 100, 54)
            sex = st.selectbox("Sex", ["Male", "Female"],
                               help="Used for the HDL (good cholesterol) goal, which differs "
                                    "for men and women.")
            bmi = st.number_input("BMI", 15.0, 60.0, 31.2, step=0.1,
                                  help="Body mass index. Pounds x 703 / (height in inches)^2, "
                                       "or use any free online BMI calculator. It is also on "
                                       "most visit summaries.")
            diabetes_duration = st.number_input("Years with diabetes", 0.0, 60.0, 6.0, step=0.5,
                                                help="Years since you were diagnosed.")
            ckd_stage = st.selectbox("Kidney stage", list(CKD_MAP), index=1,
                                     help="Ask your doctor, or match your eGFR: G1 = 90 or more, "
                                          "G2 = 60-89, G3a = 45-59, G3b = 30-44, G4 = 15-29, "
                                          "G5 = under 15. If your kidneys were never flagged, "
                                          "choose G1 or G2 by your eGFR.")
        with b:
            st.markdown("**Lab results**")
            a1c = st.number_input("A1C (%)", 4.0, 16.0, 8.4, step=0.1,
                                  help="Your 3-month average blood sugar. Look for 'A1C' or "
                                       "'HbA1c' on your lab report.")
            fasting_glucose = st.number_input("Fasting glucose (mg/dL)", 50, 500, 195,
                                              help="Blood sugar after 8+ hours without food. "
                                                   "On your lab report as 'glucose, fasting', "
                                                   "or from a home meter first thing in the morning.")
            systolic_bp = st.number_input("Blood pressure, top number", 80, 220, 138,
                                          help="The larger number, for example 138 in 138/84.")
            diastolic_bp = st.number_input("Blood pressure, bottom number", 40, 130, 84,
                                           help="The smaller number, for example 84 in 138/84.")
            egfr = st.number_input("eGFR (kidney function)", 5.0, 150.0, 71.0, step=1.0,
                                   help="Shows how well your kidneys filter. It is on your "
                                        "metabolic panel (BMP or CMP) lab report. Higher is better.")
            ldl = st.number_input("LDL cholesterol (mg/dL)", 30, 300, 118,
                                  help="The 'bad' cholesterol, from your lipid panel.")
            hdl = st.number_input("HDL cholesterol (mg/dL)", 15, 120, 42,
                                  help="The 'good' cholesterol, from your lipid panel.")
            triglycerides = st.number_input("Triglycerides (mg/dL)", 30, 800, 182,
                                            help="A type of blood fat, from your lipid panel.")
        with c:
            st.markdown("**Daily habits**")
            activity = st.selectbox("Activity level", list(ACTIVITY_MAP), index=1,
                                    help="Sedentary = mostly sitting. Lightly active = some "
                                         "walking most days. Moderately active = about 150 "
                                         "minutes of exercise a week. Very active = hard "
                                         "exercise most days.")
            dietary_pattern = st.selectbox("Eating style", list(DIET_SCORE_MAP), index=4,
                                           help="Pick the style closest to how you eat now, "
                                                "not how you want to eat.")
            restrictions_list = st.multiselect("Food restrictions", RESTRICTION_OPTIONS,
                                               help="Foods you avoid for health, allergy or "
                                                    "belief reasons. The plan will respect these.")
            caloric_intake = st.number_input("Calories per day", 800, 5000, 2100, step=50,
                                             help="A rough daily average. A food-tracking app "
                                                  "or package labels can help. A guess is fine.")
            carb_intake = st.number_input("Carbs per day (grams)", 20, 700, 240, step=5,
                                          help="Total carbohydrates in grams. It is on every "
                                               "nutrition label. A rough estimate is fine.")
            meals_per_day = st.number_input("Meals per day", 1, 8, 3,
                                            help="Count main meals. Snacks are optional.")
            fruit_veg = st.number_input("Fruit and veg servings per day", 0.0, 15.0, 2.5, step=0.5,
                                        help="One serving = 1 medium fruit, 1 cup of raw leafy "
                                             "greens, or 1/2 cup of cooked or chopped vegetables.")
            gi_awareness = st.selectbox("Know about glycemic index?", list(GI_MAP), index=1,
                                        help="Glycemic index (GI) is how fast a food raises blood "
                                             "sugar. None = never heard of it. High = you use it "
                                             "when choosing foods.")

        d, e = st.columns(2)
        with d:
            sleep_hours = st.number_input("Sleep (hours per night)", 3.0, 12.0, 6.5, step=0.5,
                                          help="Your usual night, averaged over the past month.")
            sleep_quality = st.selectbox("Sleep quality", list(SLEEP_QUALITY_MAP), index=1,
                                         help="How rested you feel on most mornings.")
            adherence = st.slider("How well do you stick to your diet? (1-10)", 1.0, 10.0, 5.5, 0.5,
                                  help="1 = you rarely follow your eating plan. "
                                       "10 = you follow it almost every day.")
            motivation = st.selectbox("Motivation", list(MOTIVATION_MAP), index=1,
                                      help="How ready you feel to change your habits right now.")
        with e:
            st.markdown("**Medicines** (check your pill bottles or pharmacy app)")
            m1, m2 = st.columns(2)
            on_metformin = m1.checkbox("Metformin", value=True,
                                       help="Also sold as Glucophage.")
            on_glp1 = m1.checkbox("GLP-1 (e.g. semaglutide)",
                                  help="Examples: semaglutide (Ozempic, Wegovy), liraglutide "
                                       "(Victoza), dulaglutide (Trulicity).")
            on_insulin = m1.checkbox("Insulin",
                                     help="Any insulin, such as Lantus, Humalog or Novolog.")
            on_sglt2 = m2.checkbox("SGLT2 inhibitor",
                                   help="Examples: empagliflozin (Jardiance), dapagliflozin "
                                        "(Farxiga), canagliflozin (Invokana).")
            on_statin = m2.checkbox("Statin", value=True,
                                    help="Cholesterol medicine, such as atorvastatin (Lipitor), "
                                         "rosuvastatin (Crestor) or simvastatin.")
            st.markdown("**Conditions** (from your doctor's diagnosis list)")
            h1, h2 = st.columns(2)
            hypertension = h1.checkbox("High blood pressure", value=True)
            dyslipidemia = h1.checkbox("High cholesterol", value=True)
            obesity = h1.checkbox("Obesity", value=True,
                                  help="Usually a BMI of 30 or higher, or as diagnosed by your doctor.")
            neuropathy = h2.checkbox("Nerve damage (neuropathy)",
                                     help="Numbness, tingling or pain in the feet or hands, "
                                          "diagnosed by a doctor.")
            retinopathy = h2.checkbox("Eye damage (retinopathy)",
                                      help="Diabetes damage to the eyes, found on an eye exam.")

        submitted = st.form_submit_button("Get my plan", type="primary")

    if submitted:
        patient = {
            "age": age, "sex": sex, "bmi": bmi, "a1c": a1c,
            "diabetes_duration": diabetes_duration, "fasting_glucose": fasting_glucose,
            "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp, "egfr": egfr,
            "ldl": ldl, "hdl": hdl, "triglycerides": triglycerides,
            "activity": activity, "dietary_pattern": dietary_pattern,
            "restrictions": ", ".join(restrictions_list) or "None",
            "caloric_intake": caloric_intake, "carb_intake": carb_intake,
            "meals_per_day": meals_per_day, "fruit_veg": fruit_veg,
            "gi_awareness": gi_awareness, "sleep_hours": sleep_hours,
            "sleep_quality": sleep_quality, "adherence": adherence,
            "motivation": motivation, "ckd_stage": ckd_stage,
            "on_metformin": on_metformin, "on_glp1": on_glp1, "on_insulin": on_insulin,
            "on_sglt2": on_sglt2, "on_statin": on_statin,
            "hypertension": hypertension, "dyslipidemia": dyslipidemia, "obesity": obesity,
            "neuropathy": neuropathy, "retinopathy": retinopathy,
        }
        if carb_intake * 4 > caloric_intake:
            st.error("Your carbs (4 calories per gram) add up to more than your daily "
                     "calories. Please check those two numbers.")
        else:
            with st.spinner("Building your plan..."):
                try:
                    st.session_state["result"] = run_pipeline(model, client, patient)
                    st.session_state["patient"] = patient
                except Exception as exc:
                    st.session_state.pop("result", None)
                    st.error(f"Something went wrong while building your plan: {exc}")

    result = st.session_state.get("result")
    if result:
        st.divider()
        render_result(result)


# ── Footer disclaimer (the only place the prototype status is mentioned) ────
st.markdown(
    '<div class="np-footer">Disclaimer: NutriPath is a prototype built on synthetic data, for education only. '
    'It does not replace advice from your doctor or diabetes educator. Your answers are sent to an AI '
    'service to write your plan and are not stored by this app.</div>', unsafe_allow_html=True)
