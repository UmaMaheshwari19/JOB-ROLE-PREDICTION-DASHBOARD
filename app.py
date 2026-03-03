import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("job_model.pkl", "rb"))

st.title("🎯 Job Role Prediction System")

st.write("Enter Student Details")

qualification = st.number_input("Qualification (Encoded Number)", min_value=0)
skills = st.number_input("Skills (Encoded Number)", min_value=0)
experience = st.number_input("Experience (Years)", min_value=0)

if st.button("Predict Job Role"):
    input_data = np.array([[qualification, skills, experience]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Job Role Code: {prediction[0]}")