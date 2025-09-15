# main.py
import pickle
import streamlit as st
import numpy as np
from os import path

st.title("Flower Classification App")

name = st.text_input("Enter your name:")
if name:
    st.write("Welcome,", name)

# model path relative to this file
MODEL_PATH = path.join(path.dirname(__file__), "model", "lr_model.pkl")

if not path.exists(MODEL_PATH):
    st.error("Model file not found. Run `python create_model.py` to create model/lr_model.pkl")
else:
    data = None
    try:
        with open(MODEL_PATH, "rb") as f:
            stored = pickle.load(f)
            model = stored["model"]
            target_names = stored["target_names"]
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
        target_names = []

    st.subheader("Enter feature values")
    sepal_length = st.slider("Sepal length (cm)", 0.0, 10.0, 5.0, step=0.1)
    sepal_width  = st.slider("Sepal width (cm)", 0.0, 10.0, 3.5, step=0.1)
    petal_length = st.slider("Petal length (cm)", 0.0, 10.0, 1.4, step=0.1)
    petal_width  = st.slider("Petal width (cm)", 0.0, 10.0, 0.2, step=0.1)

    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded.")
        else:
            X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            st.write("**Predicted class:**", target_names[pred])
            st.write("**Probabilities:**")
            for name, p in zip(target_names, probs):
                st.write(f"- {name}: {p:.3f}")
