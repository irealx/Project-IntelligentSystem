import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# load models
scaler = joblib.load("scaler.pkl")
ensemble = joblib.load("ensemble.pkl")
nn = tf.keras.models.load_model("nn.keras")

st.title("Gold Price Direction Prediction")

st.write("Predict whether gold price will go UP or DOWN tomorrow")

gold_ret = st.number_input("Gold Return (%)")
oil_ret = st.number_input("Oil Return (%)")
dxy_ret = st.number_input("Dollar Index Return (%)")
sp_ret = st.number_input("S&P500 Return (%)")

if st.button("Predict"):
    
    X = np.array([[gold_ret, oil_ret, dxy_ret, sp_ret]])
    X = scaler.transform(X)

    pred_ml = ensemble.predict(X)[0]
    pred_nn = nn.predict(X)[0][0]

    prob = (pred_ml + pred_nn) / 2

    st.subheader("Prediction Result")

    if prob > 0.5:
        st.success("Gold price will go UP tomorrow")
    else:
        st.error("Gold price will go DOWN tomorrow")

    st.write("Probability:", float(prob))