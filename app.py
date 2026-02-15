import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

st.title("ML Classification App")

model_name = st.selectbox(
    "Select Model",
    ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

uploaded_file = st.file_uploader("Upload CSV Test Data")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    with open(f"model/{model_name}.pkl", "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(data)

    st.write("Predictions:")
    st.write(predictions)

    st.write("Confusion Matrix (Demo Only):")
    st.write("Upload dataset with target column separately if needed.")
