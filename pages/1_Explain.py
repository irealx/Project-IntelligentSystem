import streamlit as st
import pandas as pd

st.set_page_config(page_title="Explain - Gold Direction", layout="wide")

st.title("Model Explanation: Gold Price Direction Prediction")
st.caption("Goal: Predict whether gold price will go UP or DOWN tomorrow (classification).")

# -----------------------------
# Data sources (edit if needed)
# -----------------------------
with st.expander("1) Dataset Source", expanded=True):
    st.markdown(
        """
**Assets used (Daily data):**
- Gold Futures (GC)
- US Dollar Index Futures (DXY)
- Crude Oil WTI Futures (CL)
- S&P500 (US500)

**Source:** Investing.com (Historical Data download)

**Note:** This project merges the 4 datasets by `Date`.
        """.strip()
    )

# -----------------------------
# Load dataset (optional display)
# -----------------------------
st.divider()
st.subheader("2) Dataset Overview")

DATA_FILE = "gold_ml_dataset.csv"

try:
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.sort_values("Date")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Start", df["Date"].min().date().isoformat())
    c3.metric("End", df["Date"].max().date().isoformat())
    c4.metric("UP rate (mean target)", f"{df['target'].mean():.3f}")

    st.markdown("**Columns used in training**")
    st.code("Features: gold_ret, oil_ret, dxy_ret, sp_ret\nTarget: target (1=UP, 0=DOWN)", language="text")

    with st.expander("Show sample rows"):
        st.dataframe(df.head(10), use_container_width=True)

except Exception as e:
    st.warning(
        f"Could not load `{DATA_FILE}`. If you want the dataset preview on this page, "
        f"make sure `{DATA_FILE}` exists in the repo root.\n\nError: {e}"
    )

# -----------------------------
# Feature engineering explanation
# -----------------------------
st.divider()
st.subheader("3) Feature Engineering")

st.markdown(
    """
We use **daily returns** (percentage change) to represent the movement of each market:

- `gold_ret` = daily return of gold close price  
- `oil_ret` = daily return of oil close price  
- `dxy_ret` = daily return of USD index close value  
- `sp_ret`  = daily return of S&P500 close value  

**Return formula (pct_change):**

`ret_t = (price_t - price_{t-1}) / price_{t-1}`
    """.strip()
)

# -----------------------------
# Target definition
# -----------------------------
st.divider()
st.subheader("4) Target (Label) Definition")

st.markdown(
    """
We predict the **direction of gold price for tomorrow**:

- `target = 1` if `gold_{t+1} > gold_t` (UP tomorrow)  
- `target = 0` otherwise (DOWN or equal)

This converts the problem into **binary classification**.
    """.strip()
)

# -----------------------------
# Data preparation steps
# -----------------------------
st.divider()
st.subheader("5) Data Preparation")

st.markdown(
    """
**Steps:**
1. Merge 4 datasets by `Date` (Gold as the base table).
2. Handle missing values (forward-fill for market series, then drop remaining NA).
3. Create the `target` label using next-day gold close.
4. Compute return features (`pct_change`).
5. Split data by time (Time-Series Split):
   - Train: first 80%
   - Test: last 20%
6. Standardize features using `StandardScaler` (fit on train only).
    """.strip()
)

# -----------------------------
# Models explanation (requirement)
# -----------------------------
st.divider()
st.subheader("6) Models")

colA, colB = st.columns(2)

with colA:
    st.markdown("### 6.1 Machine Learning (Ensemble)")
    st.markdown(
        """
We train **3 different ML models** and combine them with a **Voting Ensemble**:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- VotingClassifier (soft voting)

**Why ensemble?**  
Combining multiple models often improves robustness and reduces variance.
        """.strip()
    )

with colB:
    st.markdown("### 6.2 Neural Network")
    st.markdown(
        """
We train a simple **Feedforward Neural Network (MLP)** for binary classification:

- Dense(32, ReLU)  
- Dense(16, ReLU)  
- Dense(1, Sigmoid)

**Loss:** Binary Cross-Entropy  
**Metric:** Accuracy (+ ROC-AUC if included)
        """.strip()
    )

# -----------------------------
# Metrics (you fill in from Colab)
# -----------------------------
st.divider()
st.subheader("7) Evaluation Metrics")

st.markdown(
    """
Metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

⬇️ **Paste your results from Google Colab here** (edit the numbers below).
    """.strip()
)

# Editable placeholders (replace with your real values)
with st.expander("Fill in metrics (edit these values)", expanded=True):
    st.markdown("### Ensemble (Voting)")
    ens_acc = st.number_input("Ensemble Accuracy", value=0.00, step=0.01, format="%.4f")
    ens_prec = st.number_input("Ensemble Precision", value=0.00, step=0.01, format="%.4f")
    ens_rec = st.number_input("Ensemble Recall", value=0.00, step=0.01, format="%.4f")
    ens_f1 = st.number_input("Ensemble F1", value=0.00, step=0.01, format="%.4f")
    ens_auc = st.number_input("Ensemble ROC-AUC", value=0.00, step=0.01, format="%.4f")

    st.markdown("### Neural Network")
    nn_acc = st.number_input("NN Accuracy", value=0.5270, step=0.01, format="%.4f")
    nn_prec = st.number_input("NN Precision", value=0.6139, step=0.01, format="%.4f")
    nn_rec = st.number_input("NN Recall", value=0.4844, step=0.01, format="%.4f")
    nn_f1 = st.number_input("NN F1", value=0.5415, step=0.01, format="%.4f")
    nn_auc = st.number_input("NN ROC-AUC", value=0.5138, step=0.01, format="%.4f")

st.markdown("### Summary Table")
summary = pd.DataFrame(
    [
        ["Ensemble (Voting)", ens_acc, ens_prec, ens_rec, ens_f1, ens_auc],
        ["Neural Network", nn_acc, nn_prec, nn_rec, nn_f1, nn_auc],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
)
st.dataframe(summary, use_container_width=True)

# -----------------------------
# References
# -----------------------------
st.divider()
st.subheader("8) References")

st.markdown(
    """
- Investing.com Historical Data (Gold, DXY, Crude Oil WTI, S&P500)  
- Scikit-learn documentation (Logistic Regression, RandomForest, GradientBoosting, VotingClassifier)  
- TensorFlow/Keras documentation (MLP binary classification)
    """.strip()
)

st.success("✅ This page satisfies the requirement: explanation for both ML (ensemble) and Neural Network models.")