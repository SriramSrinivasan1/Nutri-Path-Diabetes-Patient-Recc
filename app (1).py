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
- Follow the exact section headings you are asked to use."""


def build_prompt(p, predicted_reduction, findings, safety):
    meds, conds = describe_meds_conditions(p)
    predicted_a1c = max(p["a1c"] - predicted_reduction, 0)
    findings_text = "\n".join(f"- {f}" for f in findings)
    safety_text = "\n".join(f"- {s}" for s in safety) or "- none"
    return f"""PATIENT FACTS
- Age {p['age']}, {p['sex']}, BMI {p['bmi']}, diabetes for {p['diabetes_duration']} years
- Current A1C {p['a1c']}%. Model-predicted A1C in 3 months: {predicted_a1c:.1f}%
- Eating style: {p['dietary_pattern']}. Food restrictions: {p['restrictions']}
- Medicines: {meds}. Conditions: {conds}

KEY FINDINGS (already checked by rules, use these)
{findings_text}

SAFETY CONTEXT (do not repeat, but never contradict)
{safety_text}

Write the plan using EXACTLY these sections, in this order:

## Your snapshot
2 short sentences on where they are and where they could be in 3 months.

## 3 priorities this week
Exactly 3 numbered items. Each item: one clear action, then a line starting "Why for you:"
that links to one of the key findings above.

## A sample day of eating
Breakfast, lunch, dinner, snack. Name real foods and portions that fit the restrictions.
Then give 3 simple swaps (for example "Swap X for Y").

## Your movement plan
Week 1-2, week 3-4, and week 5-12. Give minutes and days per week.

## Sleep
2 short, practical tips.

## Questions to ask your doctor
3 short questions."""


def _clean_llm_text(text):
    """Remove <think> blocks that some models add. Unclosed <think> counts as empty."""
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def build_fallback_plan(p, predicted_reduction, findings):
    """Deterministic plan used only if every LLM call fails."""
    predicted_a1c = max(p["a1c"] - predicted_reduction, 0)
    lines = [
        "## Your snapshot",
        f"Your A1C is {p['a1c']}%. With steady habits, it could reach about "
        f"{predicted_a1c:.1f}% in 3 months.",
        "",
        "## What we noticed",
    ]
    lines += [f"- {f}" for f in findings]
    lines += [
        "",
        "## 3 priorities this week",
        "1. Fill half your plate with non-starchy vegetables at lunch and dinner.",
        "2. Walk 10 minutes after your biggest meal, 5 days this week.",
        "3. Swap sweet drinks for water or unsweetened tea.",
        "",
        "## Questions to ask your doctor",
        "- Is my A1C goal right for me?",
        "- Should my medicines change as my habits change?",
        "- How often should I get my A1C checked?",
        "",
        "_The AI coach was unavailable, so this is a basic rules-based plan._",
    ]
    return "\n".join(lines)


def generate_plan(client, p, predicted_reduction, findings, safety):
    """Try the main model, then fallbacks, then the rules-based plan.
    Returns (plan_text, model_used)."""
    prompt = build_prompt(p, predicted_reduction, findings, safety)
    for model_name in [PLAN_MODEL] + FALLBACK_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.4,
            )
            text = _clean_llm_text(resp.choices[0].message.content)
            if len(text) > 200:
                return text, model_name
        except Exception:
            continue
    return build_fallback_plan(p, predicted_reduction, findings), "rules-based fallback"

# ── Pipeline: form values in, prediction + findings + plan out ──────────────
DISCLAIMER = ("Please share this plan with your doctor or diabetes educator before "
              "changing your diet, activity, or medicines.")


def run_pipeline(model, client, p):
    features = encode_patient(p)
    reduction = predict_reduction(model, features)
    predicted_a1c = round(max(p["a1c"] - reduction, 0.0), 2)
    tier, emoji = get_risk_tier(predicted_a1c)

    findings, safety = build_findings(p)
    plan_text, model_used = generate_plan(client, p, reduction, findings, safety)

    # Safety notes are added by code so the LLM can never skip or soften them
    if safety:
        plan_text += "\n\n## Safety notes for you\n" + "\n".join(f"- {s}" for s in safety)
    plan_text += f"\n\n_{DISCLAIMER}_"

    return {
        "reduction": reduction,
        "predicted_a1c": predicted_a1c,
        "risk_tier": tier,
        "risk_emoji": emoji,
        "plan": plan_text,
        "findings": findings,
        "safety": safety,
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

TIER_COLORS = {"Low": "#1D9E75", "Moderate": "#F4A623", "High": "#E8743B", "Critical": "#D0021B"}
GLYCEMIC_COLORS = {"Controlled": "#1D9E75", "Moderate": "#F4A623", "Poor": "#D0021B"}
ACTIVITY_ORDER = ["Sedentary", "Lightly active", "Moderately active", "Very active"]
MOTIVATION_ORDER = ["Low", "Moderate", "High"]

st.title("🥗 NutriPath")
st.caption("Food-as-medicine support for type 2 diabetes. Prototype built on synthetic "
           "data. Not medical advice.")

tab1, tab2 = st.tabs(["📊 Population dashboard", "🧑‍⚕️ My personal plan"])


# ── Tab 1: dashboard ────────────────────────────────────────────────────────
def avg_bar(data, group_col, order, title, color_map=None):
    summary = (data.groupby(group_col)
               .agg(avg_reduction=("predicted_a1c_reduction_3mo", "mean"),
                    patients=("predicted_a1c_reduction_3mo", "count"))
               .reset_index())
    if order:
        summary[group_col] = summary[group_col].astype(
            "category").cat.set_categories(order, ordered=True)
        summary = summary.sort_values(group_col)
    else:
        summary = summary.sort_values("avg_reduction")
    fig = px.bar(summary, x=group_col, y="avg_reduction", text="patients",
                 color=group_col, color_discrete_map=color_map, title=title,
                 labels={"avg_reduction": "Avg predicted A1C drop (points)",
                         group_col: ""})
    fig.update_traces(texttemplate="n=%{text:,}", textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(t=60, b=10))
    return fig


with tab1:
    st.subheader("Who are our patients, and what helps most?")
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

        fig1 = px.histogram(view, x="a1c_percent", color="glycemic_label", nbins=40,
                            color_discrete_map=GLYCEMIC_COLORS, barmode="overlay",
                            opacity=0.75, title="A1C distribution by blood sugar group",
                            labels={"a1c_percent": "A1C (%)", "glycemic_label": "Group"})
        fig1.add_vline(x=7.0, line_dash="dash", annotation_text="7% goal")
        st.plotly_chart(fig1)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(avg_bar(view, "diet_label", None,
                                    "Predicted drop by eating style"))
            st.plotly_chart(avg_bar(view, "motivation_label", MOTIVATION_ORDER,
                                    "Predicted drop by motivation"))
        with right:
            st.plotly_chart(avg_bar(view, "activity_label", ACTIVITY_ORDER,
                                    "Predicted drop by activity level"))
            tiers = view["risk_tier"].value_counts().reindex(
                list(TIER_COLORS)).fillna(0).reset_index()
            tiers.columns = ["risk_tier", "patients"]
            fig5 = px.bar(tiers, x="risk_tier", y="patients", color="risk_tier",
                          color_discrete_map=TIER_COLORS,
                          title="Risk tier after 3 months (predicted)",
                          labels={"risk_tier": "", "patients": "Patients"})
            fig5.update_layout(showlegend=False, margin=dict(t=60, b=10))
            st.plotly_chart(fig5)
        st.caption("Predictions come from a model trained on synthetic data, so these "
                   "charts show how the model behaves, not real-world outcomes.")


# ── Tab 2: personal plan ────────────────────────────────────────────────────
RESTRICTION_OPTIONS = ["Vegetarian", "Vegan", "Gluten-free", "Dairy-free",
                       "Nut allergy", "Halal", "Kosher", "Low-sodium"]

with tab2:
    st.subheader("Tell us about you")

    with st.expander("📖 How to use this form (read this first)", expanded=True):
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
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted A1C drop (3 months)", f"{result['reduction']:.2f} pts")
        r2.metric("Predicted A1C in 3 months", f"{result['predicted_a1c']:.1f}%")
        r3.metric("Risk tier", f"{result['risk_emoji']} {result['risk_tier']}")

        st.markdown(result["plan"])

        with st.expander("Why this plan? (what we found in your numbers)"):
            for f in result["findings"]:
                st.markdown(f"- {f}")
            st.caption(f"Plan written by: {result['model_used']}")

        st.download_button("Download my plan", data=result["plan"],
                           file_name="my_nutripath_plan.md", mime="text/markdown")
        st.caption("NutriPath is a prototype trained on synthetic data. It does not "
                   "replace advice from your care team.")
