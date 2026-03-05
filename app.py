import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="Gold Direction Prediction", layout="centered")

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.pkl")
    ensemble = joblib.load("ensemble.pkl")
    nn = tf.keras.models.load_model("nn.keras")
    return scaler, ensemble, nn

scaler, ensemble, nn = load_assets()

# -----------------------------
# Helpers
# -----------------------------
def safe_return(today: float, yesterday: float) -> float:
    """Return as ratio (e.g., 0.01 = +1%)."""
    if yesterday == 0:
        return 0.0
    return (today - yesterday) / yesterday

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# -----------------------------
# UI
# -----------------------------
st.title("Gold Price Direction Prediction")
st.write("Enter **yesterday** and **today** prices. The app will compute returns automatically.")

st.subheader("Input prices (Yesterday vs Today)")

with st.expander("Gold (GC)", expanded=True):
    gold_y = st.number_input("Gold yesterday", value=2000.0, step=1.0, format="%.2f")
    gold_t = st.number_input("Gold today", value=2005.0, step=1.0, format="%.2f")

with st.expander("Crude Oil WTI (CL)", expanded=True):
    oil_y = st.number_input("Oil yesterday", value=80.0, step=0.1, format="%.2f")
    oil_t = st.number_input("Oil today", value=80.5, step=0.1, format="%.2f")

with st.expander("US Dollar Index (DXY)", expanded=True):
    dxy_y = st.number_input("DXY yesterday", value=105.0, step=0.1, format="%.2f")
    dxy_t = st.number_input("DXY today", value=105.2, step=0.1, format="%.2f")

with st.expander("S&P500 (US500)", expanded=True):
    sp_y = st.number_input("S&P500 yesterday", value=5000.0, step=1.0, format="%.2f")
    sp_t = st.number_input("S&P500 today", value=5010.0, step=1.0, format="%.2f")

# Compute returns (ratio, not %)
gold_ret = safe_return(gold_t, gold_y)
oil_ret = safe_return(oil_t, oil_y)
dxy_ret = safe_return(dxy_t, dxy_y)
sp_ret = safe_return(sp_t, sp_y)

st.subheader("Computed returns")
c1, c2, c3, c4 = st.columns(4)
c1.metric("gold_ret", f"{gold_ret*100:.3f}%")
c2.metric("oil_ret", f"{oil_ret*100:.3f}%")
c3.metric("dxy_ret", f"{dxy_ret*100:.3f}%")
c4.metric("sp_ret", f"{sp_ret*100:.3f}%")

# Optional: clamp extreme values (protect from crazy inputs)
# Financial daily returns are usually within ~[-20%, +20%] for these assets.
X_raw = np.array([[gold_ret, oil_ret, dxy_ret, sp_ret]], dtype=float)
X_raw[0, 0] = clamp(X_raw[0, 0], -0.20, 0.20)
X_raw[0, 1] = clamp(X_raw[0, 1], -0.20, 0.20)
X_raw[0, 2] = clamp(X_raw[0, 2], -0.10, 0.10)   # DXY tends to move less
X_raw[0, 3] = clamp(X_raw[0, 3], -0.20, 0.20)

st.caption("Returns are capped to reasonable daily ranges to prevent accidental extreme inputs.")

# Predict
if st.button("Predict"):
    X = scaler.transform(X_raw)

    # Ensemble: probability of UP
    proba_ml = float(ensemble.predict_proba(X)[:, 1][0])

    # NN: probability of UP
    proba_nn = float(nn.predict(X, verbose=0).ravel()[0])

    # Average probability (simple fusion)
    proba = (proba_ml + proba_nn) / 2.0

    st.subheader("Prediction Result")
    if proba >= 0.5:
        st.success(f"Prediction: **UP tomorrow**  (probability = {proba:.3f})")
    else:
        st.error(f"Prediction: **DOWN tomorrow** (probability = {proba:.3f})")

    with st.expander("Model details"):
        st.write(f"Ensemble (Voting) probability UP: `{proba_ml:.3f}`")
        st.write(f"Neural Network probability UP: `{proba_nn:.3f}`")
        st.write("Final probability = average of two models")

    # Helpful note
    st.info("Note: This is a machine-learning prediction based on historical patterns and may be wrong.")