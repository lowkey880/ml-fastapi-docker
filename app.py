import requests
import streamlit as st

st.title("Iris Classifier")

sepal_length = st.number_input("Sepal length", value=5.1)
sepal_width = st.number_input("Sepal width", value=3.5)
petal_length = st.number_input("Petal length", value=1.4)
petal_width = st.number_input("Petal width", value=0.2)

if st.button("Predict"):
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    st.success(f"Predicted class: {response.json()['class_name']}")
