import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #141E30, #243B55);
        color: white;
    }

    label {
        color: white !important;
        font-weight: 600;
        font-size: 15px;
    }

    .stButton>button {
        background-color: #00C9A7;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Load model
with open("model.pkl", "rb") as f:
    model, le_degree, le_specialization, le_job = pickle.load(f)

# Load dataset
df = pd.read_csv("job_dataset.csv")

# Page config
st.set_page_config(page_title="Job Role Prediction", layout="centered")

# Title
st.title("🎓 Job Role Prediction Dashboard")

st.markdown("---")

# Input fields
degree = st.selectbox("Select Degree", le_degree.classes_)
specialization = st.selectbox("Select Specialization", le_specialization.classes_)
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)

# Prediction
if st.button("Predict Job"):
    deg_encoded = le_degree.transform([degree])[0]
    spec_encoded = le_specialization.transform([specialization])[0]

    prediction = model.predict([[deg_encoded, spec_encoded, cgpa]])
    result = le_job.inverse_transform(prediction)[0]

    st.success(f"Predicted Job Role: {result}")

st.markdown("---")

job_counts = df["JobRole"].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))

job_counts.plot(kind="barh", ax=ax)   # horizontal bar chart

plt.xlabel("Count")
plt.ylabel("Job Role")
plt.tight_layout()

st.pyplot(fig)