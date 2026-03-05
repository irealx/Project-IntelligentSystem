import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

st.set_page_config(page_title="Gold Direction Prediction", layout="centered")

# =============================
# THEME / CSS
# =============================
st.markdown(
    """
<style>
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"]  {
  font-family: 'Inter', sans-serif;
}

/* Hide Streamlit default footer/menu/header */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Reduce top padding */
.block-container { padding-top: 2.2rem; padding-bottom: 2.5rem; }

/* Make dividers subtle */
hr { border-color: rgba(255,255,255,0.10) !important; }

/* Page background */
.stApp {
  background: radial-gradient(1200px circle at 10% 0%, rgba(99,102,241,0.12), transparent 45%),
              radial-gradient(900px circle at 90% 10%, rgba(34,197,94,0.10), transparent 45%),
              #0b1220;
  color: #e5e7eb;
}

/* Typography */
h1, h2, h3, .stMarkdown { color: #e5e7eb; }
.small-muted { color: #9ca3af; font-size: 0.92rem; }

/* Card */
.card {
  background: rgba(17, 24, 39, 0.72);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

/* Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
  background: rgba(17, 24, 39, 0.85) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, #6366f1, #22c55e);
  color: white;
  border: 0;
  border-radius: 12px;
  padding: 0.65rem 1rem;
  font-weight: 800;
}
.stButton > button:hover { filter: brightness(1.05); }
.stButton > button:disabled { opacity: 0.45; }

/* Progress bar */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, #22c55e, #6366f1) !important;
}

/* Alerts */
div[data-testid="stAlert"] {
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(17, 24, 39, 0.75);
}

/* Metric cards */
[data-testid="stMetric"] {
  background: rgba(17, 24, 39, 0.55);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 12px 14px;
  border-radius: 14px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# Load models (cached)
# =============================
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.pkl")
    ensemble = joblib.load("ensemble.pkl")
    nn = tf.keras.models.load_model("nn.keras")
    return scaler, ensemble, nn

scaler, ensemble, nn = load_assets()

# =============================
# Helpers
# =============================
def safe_return(today: float, yesterday: float) -> float:
    """Return as ratio (0.01 = +1%)."""
    if yesterday == 0:
        return np.nan
    return (today - yesterday) / yesterday

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def validate_price_pair(yesterday: float, today: float, name: str) -> list[str]:
    errs = []
    if yesterday <= 0 or today <= 0:
        errs.append(f"{name}: prices must be > 0")
    if yesterday > 1e9 or today > 1e9:
        errs.append(f"{name}: price looks unrealistically large")
    return errs

def predict_proba_up(X_raw: np.ndarray):
    """X_raw is shape (1,4) in return-ratio units."""
    X = scaler.transform(X_raw)
    proba_ml = float(ensemble.predict_proba(X)[:, 1][0])
    proba_nn = float(nn.predict(X, verbose=0).ravel()[0])
    proba = (proba_ml + proba_nn) / 2.0
    return proba, proba_ml, proba_nn

def pct(x: float) -> str:
    return f"{x*100:.3f}%"

# =============================
# UI Header
# =============================
st.title("Gold Price Direction Prediction")

st.markdown(
    """
<div class="card" style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
  <div>
    <div style="font-size:1.25rem;font-weight:900;letter-spacing:-0.02em;">Gold Direction Predictor</div>
    <div class="small-muted" style="margin-top:6px;">
      Predict tomorrow's <b>UP/DOWN</b> using <b>Ensemble ML</b> + <b>Neural Network</b>.
      <br/>Inputs are converted to <b>daily returns</b> before prediction.
    </div>
  </div>
  <div style="
      padding:8px 12px;
      border-radius:999px;
      background: rgba(99,102,241,0.18);
      border: 1px solid rgba(99,102,241,0.35);
      color:#c7d2fe;
      font-weight:800;
      font-size:0.85rem;">
    Deployed • Streamlit
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# =============================
# Mode selector + actions
# =============================
mode = st.radio(
    "Input mode",
    ["Price mode (Yesterday vs Today)", "Return mode (direct %)"],
    horizontal=True,
)

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    use_example = st.button("Use example")
with colB:
    reset = st.button("Reset")
with colC:
    st.write("Tip: **Price mode** is more user-friendly. Return mode is for advanced users.")

EX = {
    "gold_y": 2000.0, "gold_t": 2005.0,
    "oil_y": 80.0, "oil_t": 80.5,
    "dxy_y": 105.0, "dxy_t": 105.2,
    "sp_y": 5000.0, "sp_t": 5010.0,
    "gold_r": 0.25, "oil_r": 0.30, "dxy_r": 0.10, "sp_r": 0.20
}

if reset:
    # clear only our keys
    for k in list(EX.keys()):
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

if use_example:
    for k, v in EX.items():
        st.session_state[k] = v
    st.rerun()

st.divider()

# =============================
# INPUTS
# =============================
errs = []
X_raw = None  # (1,4) return ratios
gold_ret = oil_ret = dxy_ret = sp_ret = np.nan

if mode.startswith("Price mode"):
    st.subheader("Enter prices (Yesterday vs Today)")
    st.write("Units are shown in the labels. The app calculates daily returns automatically.")

    with st.expander("Gold (GC) — USD/oz", expanded=True):
        gold_y = st.number_input(
            "Gold yesterday (USD/oz)",
            key="gold_y",
            value=st.session_state.get("gold_y", EX["gold_y"]),
            step=1.0,
            format="%.2f",
            help="Gold futures price in USD per troy ounce (USD/oz)."
        )
        gold_t = st.number_input(
            "Gold today (USD/oz)",
            key="gold_t",
            value=st.session_state.get("gold_t", EX["gold_t"]),
            step=1.0,
            format="%.2f",
            help="Gold futures price in USD per troy ounce (USD/oz)."
        )

    with st.expander("Crude Oil WTI (CL) — USD/bbl", expanded=True):
        oil_y = st.number_input(
            "Oil yesterday (USD/bbl)",
            key="oil_y",
            value=st.session_state.get("oil_y", EX["oil_y"]),
            step=0.1,
            format="%.2f",
            help="WTI crude oil price in USD per barrel (USD/bbl)."
        )
        oil_t = st.number_input(
            "Oil today (USD/bbl)",
            key="oil_t",
            value=st.session_state.get("oil_t", EX["oil_t"]),
            step=0.1,
            format="%.2f",
            help="WTI crude oil price in USD per barrel (USD/bbl)."
        )

    with st.expander("US Dollar Index (DXY) — index", expanded=True):
        dxy_y = st.number_input(
            "DXY yesterday (index)",
            key="dxy_y",
            value=st.session_state.get("dxy_y", EX["dxy_y"]),
            step=0.1,
            format="%.2f",
            help="US Dollar Index (DXY) in index points."
        )
        dxy_t = st.number_input(
            "DXY today (index)",
            key="dxy_t",
            value=st.session_state.get("dxy_t", EX["dxy_t"]),
            step=0.1,
            format="%.2f",
            help="US Dollar Index (DXY) in index points."
        )

    with st.expander("S&P500 (US500) — index", expanded=True):
        sp_y = st.number_input(
            "S&P500 yesterday (index)",
            key="sp_y",
            value=st.session_state.get("sp_y", EX["sp_y"]),
            step=1.0,
            format="%.2f",
            help="S&P 500 (US500) in index points."
        )
        sp_t = st.number_input(
            "S&P500 today (index)",
            key="sp_t",
            value=st.session_state.get("sp_t", EX["sp_t"]),
            step=1.0,
            format="%.2f",
            help="S&P 500 (US500) in index points."
        )

    errs += validate_price_pair(gold_y, gold_t, "Gold")
    errs += validate_price_pair(oil_y, oil_t, "Oil")
    errs += validate_price_pair(dxy_y, dxy_t, "DXY")
    errs += validate_price_pair(sp_y, sp_t, "S&P500")

    gold_ret = safe_return(gold_t, gold_y)
    oil_ret  = safe_return(oil_t, oil_y)
    dxy_ret  = safe_return(dxy_t, dxy_y)
    sp_ret   = safe_return(sp_t, sp_y)

else:
    st.subheader("Enter daily returns directly (%)")
    st.write("Example: enter **1.25** for +1.25% (the app converts to ratio internally).")

    c1, c2 = st.columns(2)
    with c1:
        gold_r = st.number_input("Gold daily return (%)", key="gold_r",
                                 value=st.session_state.get("gold_r", EX["gold_r"]),
                                 step=0.05, format="%.3f")
        oil_r  = st.number_input("Oil daily return (%)", key="oil_r",
                                 value=st.session_state.get("oil_r", EX["oil_r"]),
                                 step=0.05, format="%.3f")
    with c2:
        dxy_r  = st.number_input("DXY daily return (%)", key="dxy_r",
                                 value=st.session_state.get("dxy_r", EX["dxy_r"]),
                                 step=0.02, format="%.3f")
        sp_r   = st.number_input("S&P500 daily return (%)", key="sp_r",
                                 value=st.session_state.get("sp_r", EX["sp_r"]),
                                 step=0.05, format="%.3f")

    gold_ret = gold_r / 100.0
    oil_ret  = oil_r / 100.0
    dxy_ret  = dxy_r / 100.0
    sp_ret   = sp_r / 100.0

# Build X
X_raw = np.array([[gold_ret, oil_ret, dxy_ret, sp_ret]], dtype=float)

# validate returns
if np.any(~np.isfinite(X_raw)):
    errs.append("Some returns are invalid (check inputs).")

# Clamp extreme values (protect from accidental crazy inputs)
if len(errs) == 0:
    X_raw[0, 0] = clamp(float(X_raw[0, 0]), -0.20, 0.20)
    X_raw[0, 1] = clamp(float(X_raw[0, 1]), -0.20, 0.20)
    X_raw[0, 2] = clamp(float(X_raw[0, 2]), -0.10, 0.10)
    X_raw[0, 3] = clamp(float(X_raw[0, 3]), -0.20, 0.20)

# =============================
# Show computed returns + features table
# =============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Computed features (daily returns)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("gold_ret", pct(float(X_raw[0, 0])) if np.isfinite(X_raw[0, 0]) else "—")
m2.metric("oil_ret",  pct(float(X_raw[0, 1])) if np.isfinite(X_raw[0, 1]) else "—")
m3.metric("dxy_ret",  pct(float(X_raw[0, 2])) if np.isfinite(X_raw[0, 2]) else "—")
m4.metric("sp_ret",   pct(float(X_raw[0, 3])) if np.isfinite(X_raw[0, 3]) else "—")

feat_df = pd.DataFrame(
    {
        "Feature": ["gold_ret", "oil_ret", "dxy_ret", "sp_ret"],
        "Value (ratio)": [float(X_raw[0, 0]), float(X_raw[0, 1]), float(X_raw[0, 2]), float(X_raw[0, 3])],
        "Value (%)": [pct(float(X_raw[0, 0])), pct(float(X_raw[0, 1])), pct(float(X_raw[0, 2])), pct(float(X_raw[0, 3]))],
    }
)
with st.expander("Show feature table"):
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =============================
# Errors (if any)
# =============================
if errs:
    st.error("Please fix the following issues:")
    for e in errs:
        st.write(f"- {e}")

# =============================
# Predict
# =============================
btn = st.button("Predict", disabled=bool(errs))

if btn and not errs:
    proba, proba_ml, proba_nn = predict_proba_up(X_raw)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    st.progress(float(proba))
    st.write(f"**Final probability (UP)** = `{proba*100:.1f}%`")

    if proba >= 0.5:
        st.success("Prediction: **UP tomorrow**")
        st.caption("Interpretation: the model sees market signals leaning bullish for gold (tomorrow).")
    else:
        st.error("Prediction: **DOWN tomorrow**")
        st.caption("Interpretation: the model sees market signals leaning bearish for gold (tomorrow).")

    with st.expander("Model breakdown (details)"):
        st.write(f"Ensemble (Voting) P(UP): `{proba_ml:.3f}`")
        st.write(f"Neural Network P(UP): `{proba_nn:.3f}`")
        st.write("Final probability = average of Ensemble and Neural Network")

    st.info("Note: This is a prediction from historical patterns. Financial markets are noisy and predictions can be wrong.")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

with st.expander("How this app works (quick explanation)"):
    st.markdown(
        """
- You provide **today and yesterday prices** (or daily returns).
- The app computes **daily returns**:
  - `ret = (today - yesterday) / yesterday`
- Features used by the model:
  - `gold_ret`, `oil_ret`, `dxy_ret`, `sp_ret`
- Models:
  - **Machine Learning Ensemble** (Voting of LR + RF + GB)
  - **Neural Network** (MLP)
- Output:
  - UP/DOWN prediction for tomorrow with probability.
        """.strip()
    )