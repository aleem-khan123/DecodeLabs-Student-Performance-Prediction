import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("student_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.title-box {
    background: linear-gradient(90deg, #1f77b4, #00b4d8);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
}

.info-card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #1f77b4;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.result-card {
    background: linear-gradient(90deg, #e3f2fd, #ffffff);
    padding: 25px;
    border-radius: 15px;
    border: 2px solid #1f77b4;
    text-align: center;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #555;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="title-box">
    <h1> Student Performance Prediction System</h1>
    <p>Predict student exam score using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
This application predicts a student's <b>Exam Score</b> based on academic and lifestyle factors such as study hours, attendance, sleep hours, internet usage, assignments completed, and previous score.
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Student Information")

study_hours = st.sidebar.slider("Study Hours", 1, 11, 6)
attendance = st.sidebar.slider("Attendance (%)", 40, 100, 70)
sleep_hours = st.sidebar.slider("Sleep Hours", 4, 9, 7)
internet_usage = st.sidebar.slider("Internet Usage (Hours)", 1, 11, 6)
assignments_completed = st.sidebar.slider("Assignments Completed", 0, 20, 10)
previous_score = st.sidebar.slider("Previous Score", 35, 95, 65)

# Input Data
input_data = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "sleep_hours": [sleep_hours],
    "internet_usage": [internet_usage],
    "assignments_completed": [assignments_completed],
    "previous_score": [previous_score]
})

# Input Summary
st.subheader("Input Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Study Hours", study_hours)
    st.metric("Attendance", f"{attendance}%")

with col2:
    st.metric("Sleep Hours", sleep_hours)
    st.metric("Internet Usage", internet_usage)

with col3:
    st.metric("Assignments", assignments_completed)
    st.metric("Previous Score", previous_score)

st.markdown("---")

# Prediction
if st.button("Predict Exam Score"):
    prediction = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="result-card">
        <h2>Predicted Exam Score</h2>
        <h1>{prediction:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

    if prediction >= 90:
        st.success("Performance Level: Excellent")
        st.progress(95)

    elif prediction >= 75:
        st.info("Performance Level: Good")
        st.progress(80)

    elif prediction >= 60:
        st.warning("Performance Level: Average")
        st.progress(65)

    else:
        st.error("Performance Level: Needs Improvement")
        st.progress(45)

st.markdown("---")

# About Project
st.subheader("About This Project")

st.markdown("""
<div class="info-card">
<b>Project Type:</b> Machine Learning Web Application<br>
<b>Algorithm Used:</b> Linear Regression<br>
<b>Target Variable:</b> Exam Score<br>
<b>Dataset Size:</b> 10,000 student records
</div>
""", unsafe_allow_html=True)

st.subheader("Model Performance")

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("MAE", "6.92")

with col5:
    st.metric("RMSE", "8.79")

with col6:
    st.metric("R² Score", "0.66")

st.markdown("---")

st.markdown("""
<div class="footer">
Developed by Aleem Shoukat | Decode Labs Data Science Internship Project
</div>
""", unsafe_allow_html=True)