# app.py – Uganda Internet Access Predictor (XGBoost)
# Deployed Model: XGBoost – Best Performer (ROC-AUC: 0.879, F1: 0.52)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Uganda Internet Access Predictor",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-yes {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 1rem;
        padding: 1.5rem;
        text-align: center;
    }
    .prediction-no {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 1rem;
        padding: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🌍 Uganda Internet Access Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Based on 2014 Uganda Census Data | XGBoost Model (Best Performer: ROC-AUC 0.879)</div>', unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('xgb_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info("Please ensure xgb_model.pkl, scaler.pkl, and feature_columns.pkl are in the same directory.")
        return None, None, None

model, scaler, feature_columns = load_artifacts()

if model is None:
    st.stop()

# Sidebar inputs
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Flag_of_Uganda.svg/1200px-Flag_of_Uganda.svg.png", width=150)
    st.markdown("## 📊 Household Information")
    st.markdown("---")
    
    # Demographics
    st.markdown("### 👤 Demographics")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=25)
    sex = st.selectbox("Sex", ["Male", "Female"], index=0)
    sex_value = 1 if sex == "Male" else 2
    
    # Education
    st.markdown("### 🎓 Education")
    literacy = st.selectbox("Literacy", ["Literate", "Not Literate"], index=0)
    literacy_value = 1 if literacy == "Literate" else 2
    grade = st.slider("Highest Grade Completed", min_value=0, max_value=20, value=7)
    attending = st.selectbox("Currently Attending School", ["No", "Yes"], index=0)
    attending_value = 2 if attending == "No" else 1
    
    # Location
    st.markdown("### 🗺️ Location")
    rururb = st.selectbox("Area Type", ["Urban", "Rural"], index=1)
    rururb_value = 1 if rururb == "Urban" else 2
    region = st.selectbox("Region", list(range(1, 16)), index=0)
    
    # Assets
    st.markdown("### 📱 Assets")
    phone = st.selectbox("Mobile Phone", ["No", "Yes"], index=1)
    phone_value = 1 if phone == "Yes" else 0
    computer = st.selectbox("Computer", ["No", "Yes"], index=1)
    computer_value = 1 if computer == "Yes" else 0
    television = st.selectbox("Television", ["No", "Yes"], index=1)
    television_value = 1 if television == "Yes" else 0
    radio = st.selectbox("Radio", ["No", "Yes"], index=1)
    radio_value = 1 if radio == "Yes" else 0
    
    # Infrastructure
    st.markdown("### 🏠 Infrastructure")
    energy = st.selectbox("Energy Source", ["Grid", "Solar", "Generator", "Other"], index=0)
    energy_map = {"Grid": 1, "Solar": 2, "Generator": 3, "Other": 4}
    energy_value = energy_map[energy]
    
    water = st.selectbox("Water Source", ["Piped", "Borehole", "Well", "Spring", "Other"], index=0)
    water_map = {"Piped": 1, "Borehole": 2, "Well": 3, "Spring": 4, "Other": 5}
    water_value = water_map[water]
    
    toilet = st.selectbox("Toilet", ["Flush", "Pit Latrine", "VIP", "None"], index=1)
    toilet_map = {"Flush": 1, "Pit Latrine": 2, "VIP": 3, "None": 4}
    toilet_value = toilet_map[toilet]
    
    bank = st.selectbox("Bank Account", ["No", "Yes"], index=1)
    bank_value = 1 if bank == "Yes" else 0
    
    st.markdown("---")

# Function to preprocess inputs into model format
def preprocess_inputs():
    """Convert user inputs to the exact feature format expected by the model"""
    
    # Create base dataframe with raw values
    raw_df = pd.DataFrame([{
        'age': age,
        'sex': sex_value,
        'literacy': literacy_value,
        'Grade': grade,
        'attending': attending_value,
        'rururb': rururb_value,
        'phone': phone_value,
        'computer': computer_value,
        'television': television_value,
        'radio': radio_value,
        'energysource': energy_value,
        'waterdrinking': water_value,
        'toilet': toilet_value,
        'bank_account': bank_value,
        'Region15': region
    }])
    
    # One-hot encode (same as training)
    raw_df = pd.get_dummies(raw_df, columns=['sex', 'rururb', 'Region15'], 
                            prefix=['sex', 'rururb', 'Region15'])
    
    # Label encode ordinal columns
    if 'literacy' in raw_df.columns:
        raw_df['literacy_encoded'] = raw_df['literacy']
        raw_df = raw_df.drop('literacy', axis=1)
    if 'attending' in raw_df.columns:
        raw_df['attending_encoded'] = raw_df['attending']
        raw_df = raw_df.drop('attending', axis=1)
    
    # Add missing columns with 0
    for col in feature_columns:
        if col not in raw_df.columns:
            raw_df[col] = 0
    
    # Reorder to match training
    raw_df = raw_df[feature_columns]
    
    return raw_df.values.astype(np.float32)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📊 Model Performance")
    st.markdown("""
    | Metric | XGBoost Value |
    |--------|---------------|
    | **ROC-AUC** | 0.879 |
    | **Accuracy** | 90% |
    | **Recall (Yes)** | 60% |
    | **Precision (Yes)** | 45% |
    | **F1-Score (Yes)** | 0.52 |
    """)
    
    st.markdown("## 📈 Key Predictors (Based on Model)")
    st.markdown("""
    1. **Phone ownership** – strongest predictor
    2. **Education (Grade)** – higher education = more internet
    3. **Region** – geographic location matters
    4. **Urban/Rural status** – urban areas better connected
    5. **Age** – younger people more likely connected
    """)

with col2:
    st.markdown("## 🎯 Prediction")
    
    if st.button("🔮 Predict Internet Access", type="primary", use_container_width=True):
        try:
            # Preprocess inputs
            input_features = preprocess_inputs()
            
            # Scale
            input_scaled = scaler.transform(input_features)
            
            # Predict
            proba = model.predict_proba(input_scaled)[0][1]
            prediction = "Has Internet" if proba > 0.5 else "No Internet"
            
            # Display result
            if prediction == "Has Internet":
                st.markdown(f"""
                <div class="prediction-yes">
                    <h1 style="color: #10B981; margin:0;">✅ HAS INTERNET ACCESS</h1>
                    <p style="font-size: 2rem; margin: 0.5rem;">Probability: {proba:.1%}</p>
                    <p>This household is likely connected to the internet.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-no">
                    <h1 style="color: #EF4444; margin:0;">❌ NO INTERNET ACCESS</h1>
                    <p style="font-size: 2rem; margin: 0.5rem;">Probability: {(1-proba):.1%}</p>
                    <p>This household is likely not connected to the internet.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Internet Access Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, 30], 'color': '#FEE2E2'},
                        {'range': [30, 70], 'color': '#FEF3C7'},
                        {'range': [70, 100], 'color': '#D1FAE5'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': proba * 100
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Please ensure all preprocessing steps match the training pipeline.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
    <p>Based on Uganda National Population and Housing Census 2014 | Model: XGBoost (Best Performer)</p>
    <p>For research and policy planning purposes only.</p>
</div>
""", unsafe_allow_html=True)