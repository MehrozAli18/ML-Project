# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Streamlit UI Development
#  File: streamlit_app.py
#  Task: CTG Fetal State Classification
# ══════════════════════════════════════════════════════════════════════════════


# ── 4.1 IMPORT LIBRARIES ──────────────────────────────────────────────────────
import streamlit as st          # Main UI framework — builds the web app
import joblib                   # Loads the saved trained model (.pkl file)
import numpy as np                # Numerical operations on input data
import pandas as pd             # Organising user input into a DataFrame


# ── 4.2 SET PAGE TITLE AND DOCUMENT TITLE ────────────────────────────────────
# page_title  → appears on the browser tab
# page_icon   → emoji shown on the browser tab
# layout      → 'wide' uses full screen width

st.set_page_config(
    page_title="CTG Fetal State Classifier",
    page_icon="🏥",
    layout="wide"
)

# Main heading shown on the app page
st.title("🏥 Cardiotocography (CTG) — Fetal State Classifier")
st.markdown("#### Predict fetal health state based on CTG signal readings")
st.markdown("---")


# ── 4.3 LOAD MODEL ────────────────────────────────────────────────────────────
# Load the serialized model and scaler saved during Step 3 (model.ipynb).
# @st.cache_resource → loads the model ONCE and reuses it,
#                      so the app doesn't reload the model on every click.

@st.cache_resource
def load_model():
    model  = joblib.load('rf_best_model.pkl')   # Load trained Random Forest
    scaler = joblib.load('scaler.pkl')           # Load fitted StandardScaler
    return model, scaler

model, scaler = load_model()
st.success("✅ Model loaded successfully!")


# ── 4.4 GET INPUT VIA STREAMLIT INPUT FUNCTIONS ───────────────────────────────
# We collect all 21 CTG feature values from the user.
# Sliders are used for numerical values — each slider has:
#   - Label         → feature name
#   - min_value     → lowest allowed value
#   - max_value     → highest allowed value
#   - value         → default value (typical/median value)
#   - step          → increment size

st.markdown("### 📋 Enter Patient CTG Readings")
st.markdown("Adjust the sliders below to input the patient's CTG signal values:")
st.markdown("")

# Split inputs across 3 columns for a clean layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**❤️ Heart Rate Features**")
    LB      = st.slider("LB — FHR Baseline (bpm)",     min_value=50,   max_value=200,  value=120, step=1)
    AC      = st.slider("AC — Accelerations/sec",       min_value=0.0,  max_value=0.02, value=0.003, step=0.001, format="%.3f")
    FM      = st.slider("FM — Fetal Movements/sec",     min_value=0.0,  max_value=0.6,  value=0.0,   step=0.001, format="%.3f")
    UC      = st.slider("UC — Uterine Contractions/sec",min_value=0.0,  max_value=0.02, value=0.003, step=0.001, format="%.3f")
    DL      = st.slider("DL — Light Decelerations",     min_value=0.0,  max_value=0.02, value=0.001, step=0.001, format="%.3f")
    DS      = st.slider("DS — Severe Decelerations",    min_value=0.0,  max_value=0.01, value=0.0,   step=0.001, format="%.3f")
    DP      = st.slider("DP — Prolongued Decelerations",min_value=0.0,  max_value=0.01, value=0.0,   step=0.001, format="%.3f")

with col2:
    st.markdown("**📈 Variability Features**")
    ASTV    = st.slider("ASTV — % Abnormal Short-Term Variability", min_value=0,   max_value=100,  value=73,  step=1)
    MSTV    = st.slider("MSTV — Mean Short-Term Variability",       min_value=0.0, max_value=20.0, value=0.5, step=0.1, format="%.1f")
    ALTV    = st.slider("ALTV — % Abnormal Long-Term Variability",  min_value=0,   max_value=100,  value=43,  step=1)
    MLTV    = st.slider("MLTV — Mean Long-Term Variability",        min_value=0.0, max_value=50.0, value=2.4, step=0.1, format="%.1f")

with col3:
    st.markdown("**📊 Histogram Features**")
    Width   = st.slider("Width — Width of FHR Histogram",    min_value=0,   max_value=200,  value=64,  step=1)
    Min     = st.slider("Min — Minimum FHR",                 min_value=50,  max_value=200,  value=62,  step=1)
    Max     = st.slider("Max — Maximum FHR",                 min_value=50,  max_value=300,  value=126, step=1)
    Nmax    = st.slider("Nmax — Number of Histogram Peaks",  min_value=0,   max_value=20,   value=4,   step=1)
    Nzeros  = st.slider("Nzeros — Number of Zeros",          min_value=0,   max_value=20,   value=0,   step=1)
    Mode    = st.slider("Mode — Mode of FHR Histogram",      min_value=50,  max_value=200,  value=120, step=1)
    Mean    = st.slider("Mean — Mean FHR",                   min_value=50,  max_value=200,  value=137, step=1)
    Median  = st.slider("Median — Median FHR",               min_value=50,  max_value=200,  value=121, step=1)
    Variance= st.slider("Variance — Variance of FHR",        min_value=0,   max_value=300,  value=73,  step=1)
    Tendency= st.slider("Tendency — Trend of FHR",           min_value=-1,  max_value=1,    value=1,   step=1)


# ── 4.5 MAKE PREDICTIONS AND DISPLAY RESULTS ──────────────────────────────────
# When the user clicks Predict, we:
#   1. Collect all slider values into a DataFrame
#   2. Scale them using the loaded scaler
#   3. Pass to the model for prediction
#   4. Display the result clearly

st.markdown("---")

if st.button("🔍 Predict Fetal State", use_container_width=True):

    # Step 1 — Organise all inputs into a single row DataFrame
    input_data = pd.DataFrame([{
        'LB': LB, 'AC': AC, 'FM': FM, 'UC': UC,
        'DL': DL, 'DS': DS, 'DP': DP,
        'ASTV': ASTV, 'MSTV': MSTV, 'ALTV': ALTV, 'MLTV': MLTV,
        'Width': Width, 'Min': Min, 'Max': Max,
        'Nmax': Nmax, 'Nzeros': Nzeros, 'Mode': Mode,
        'Mean': Mean, 'Median': Median,
        'Variance': Variance, 'Tendency': Tendency
    }])

    # Step 2 — Scale the input using the same scaler from training
    # IMPORTANT: Always use the saved scaler — never fit a new one on user input
    input_scaled = scaler.transform(input_data)

    # Step 3 — Get prediction and confidence probabilities
    prediction   = model.predict(input_scaled)[0]           # Class: 1, 2, or 3
    probabilities= model.predict_proba(input_scaled)[0]     # Confidence per class
    confidence   = probabilities.max() * 100                # Highest confidence %

    # Step 4 — Display results
    st.markdown("### 🏆 Prediction Result")
    result_col1, result_col2, result_col3 = st.columns(3)

    # Map class numbers to meaningful labels and messages
    if prediction == 1:
        result_col1.success("✅ **Normal**")
        st.success(f"**Fetal State: NORMAL** — The CTG readings appear healthy. Confidence: {confidence:.1f}%")

    elif prediction == 2:
        result_col1.warning("⚠️ **Suspect**")
        st.warning(f"**Fetal State: SUSPECT** — The readings show some irregularities. Further monitoring recommended. Confidence: {confidence:.1f}%")

    elif prediction == 3:
        result_col1.error("🚨 **Pathologic**")
        st.error(f"**Fetal State: PATHOLOGIC** — The readings indicate a high-risk condition. Immediate medical attention required. Confidence: {confidence:.1f}%")

    # Show probability breakdown for all 3 classes
    st.markdown("#### 📊 Confidence Breakdown")
    prob_df = pd.DataFrame({
        'Fetal State': ['Normal (1)', 'Suspect (2)', 'Pathologic (3)'],
        'Confidence (%)': [round(p * 100, 2) for p in probabilities]
    })
    st.dataframe(prob_df, use_container_width=True)

    # Show the raw input values the user entered
    st.markdown("#### 📋 Input Values Summary")
    st.dataframe(input_data, use_container_width=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "⚠️ *This tool is for educational purposes only and should not replace professional medical diagnosis.*"
)