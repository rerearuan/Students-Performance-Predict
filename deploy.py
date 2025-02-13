import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("🎓 Sistem Prediksi Dropout Mahasiswa")
st.write("""
Aplikasi ini membantu memprediksi risiko dropout mahasiswa berdasarkan berbagai indikator akademik dan non-akademik.
""")

# Create tabs
tab1, = st.tabs(["Prediksi Individual"]) 


with tab1:
    st.header("Prediksi Individual")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        sem1_grade = st.number_input("Nilai Semester 1", 0.0, 20.0, 10.0)
        sem2_grade = st.number_input("Nilai Semester 2", 0.0, 20.0, 10.0)
        first_sem_perf = st.slider("Performa Semester 1 (%)", 0.0, 100.0, 50.0) / 100
        second_sem_perf = st.slider("Performa Semester 2 (%)", 0.0, 100.0, 50.0) / 100
        avg_grade = st.number_input("Rata-rata Nilai", 0.0, 20.0, 10.0)

    with col2:
        approval_ratio = st.slider("Rasio Kelulusan Total (%)", 0.0, 100.0, 50.0) / 100
        perf_change = second_sem_perf - first_sem_perf
        financial_diff = st.checkbox("Ada Kesulitan Finansial")
        academic_risk = approval_ratio * (1 + int(financial_diff))
        age = st.number_input("Usia saat Masuk", 16, 60, 18)

    # Create prediction button
    if st.button("Prediksi Risiko Dropout"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Curricular_units_2nd_sem_grade': [sem2_grade],
            'Curricular_units_1st_sem_grade': [sem1_grade],
            'first_sem_performance': [first_sem_perf],
            'second_sem_performance': [second_sem_perf],
            'avg_grade': [avg_grade],
            'total_approval_ratio': [approval_ratio],
            'performance_change': [perf_change],
            'financial_difficulty': [int(financial_diff)],
            'academic_financial_risk': [academic_risk],
            'Age_at_enrollment': [age]
        })

        # Load model and scaler (you need to save these first)
        try:
            model = pickle.load(open('best_model.pkl', 'rb'))
            scaler = pickle.load(open('scaler_model.pkl', 'rb'))
            
            # Scale input data
            input_scaled = scaler.transform(input_data)
            
            # Get prediction probability
            dropout_prob = model.predict_proba(input_scaled)[0][1]
            
            # Determine risk level
            if dropout_prob < 0.3:
                risk_level = "Rendah"
                color = "green"
            elif dropout_prob < 0.6:
                risk_level = "Sedang"
                color = "orange"
            else:
                risk_level = "Tinggi"
                color = "red"
            
            # Display results
            st.markdown(f"""
            ### Hasil Prediksi
            - Probabilitas Dropout: **{dropout_prob:.1%}**
            - Tingkat Risiko: <span style='color:{color}'><strong>{risk_level}</strong></span>
            """, unsafe_allow_html=True)
            
        except FileNotFoundError:
            st.error("Model belum di-upload. Silakan upload model terlebih dahulu.")
