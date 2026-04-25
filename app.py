import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Uganda Internet Predictor - LightGBM", layout="wide")
st.title("🌍 Uganda Internet Access Predictor")
st.markdown("Based on 2014 Uganda Census Data | **LightGBM Model** (Best Recall: 67%)")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .prediction-yes {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 1rem;
        padding: 1rem;
        text-align: center;
    }
    .prediction-no {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 1rem;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load('lgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_artifacts()
    st.success("✅ LightGBM Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar inputs
st.sidebar.header("📊 Household Information")

age = st.sidebar.number_input("Age", 0, 120, 25)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex_val = 1 if sex == "Male" else 2

literacy = st.sidebar.selectbox("Literacy", ["Literate", "Not Literate"])
literacy_val = 1 if literacy == "Literate" else 2

grade = st.sidebar.slider("Grade Completed", 0, 20, 7)
attending = st.sidebar.selectbox("Attending School", ["No", "Yes"])
attending_val = 2 if attending == "No" else 1

rururb = st.sidebar.selectbox("Area Type", ["Urban", "Rural"])
rururb_val = 1 if rururb == "Urban" else 2

region = st.sidebar.selectbox("Region", list(range(1, 16)), index=0)

phone = st.sidebar.selectbox("Mobile Phone", ["No", "Yes"])
phone_val = 1 if phone == "Yes" else 0

computer = st.sidebar.selectbox("Computer", ["No", "Yes"])
computer_val = 1 if computer == "Yes" else 0

television = st.sidebar.selectbox("Television", ["No", "Yes"])
television_val = 1 if television == "Yes" else 0

radio = st.sidebar.selectbox("Radio", ["No", "Yes"])
radio_val = 1 if radio == "Yes" else 0

energy = st.sidebar.selectbox("Energy Source", ["Grid", "Solar", "Generator", "Other"])
energy_map = {"Grid": 1, "Solar": 2, "Generator": 3, "Other": 4}
energy_val = energy_map[energy]

water = st.sidebar.selectbox("Water Source", ["Piped", "Borehole", "Well", "Spring", "Other"])
water_map = {"Piped": 1, "Borehole": 2, "Well": 3, "Spring": 4, "Other": 5}
water_val = water_map[water]

toilet = st.sidebar.selectbox("Toilet", ["Flush", "Pit Latrine", "VIP", "None"])
toilet_map = {"Flush": 1, "Pit Latrine": 2, "VIP": 3, "None": 4}
toilet_val = toilet_map[toilet]

bank = st.sidebar.selectbox("Bank Account", ["No", "Yes"])
bank_val = 1 if bank == "Yes" else 0

def preprocess():
    raw_df = pd.DataFrame([{
        'age': age, 'sex': sex_val, 'literacy': literacy_val, 'Grade': grade,
        'attending': attending_val, 'rururb': rururb_val, 'phone': phone_val,
        'computer': computer_val, 'television': television_val, 'radio': radio_val,
        'energysource': energy_val, 'waterdrinking': water_val, 'toilet': toilet_val,
        'bank_account': bank_val, 'Region15': region
    }])
    
    raw_df = pd.get_dummies(raw_df, columns=['sex', 'rururb', 'Region15'], 
                            prefix=['sex', 'rururb', 'Region15'])
    
    if 'literacy' in raw_df.columns:
        raw_df['literacy_encoded'] = raw_df['literacy']
        raw_df = raw_df.drop('literacy', axis=1)
    if 'attending' in raw_df.columns:
        raw_df['attending_encoded'] = raw_df['attending']
        raw_df = raw_df.drop('attending', axis=1)
    
    for col in feature_columns:
        if col not in raw_df.columns:
            raw_df[col] = 0
    
    raw_df = raw_df[feature_columns]
    return raw_df.values.astype(np.float32)

st.markdown("## 🎯 Prediction")

if st.button("🔮 Predict Internet Access", type="primary", use_container_width=True):
    try:
        features = preprocess()
        scaled = scaler.transform(features)
        proba = model.predict_proba(scaled)[0][1]
        
        if proba > 0.5:
            st.markdown(f"""
            <div class="prediction-yes">
                <h2 style="color: #10B981;">✅ HAS INTERNET ACCESS</h2>
                <h3>Probability: {proba:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-no">
                <h2 style="color: #EF4444;">❌ NO INTERNET ACCESS</h2>
                <h3>Probability: {(1-proba):.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.progress(proba)
        
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.markdown("📊 **LightGBM Performance:** ROC-AUC: 0.878 | Recall: 67% | F1: 0.49")
st.caption("LightGBM has the highest recall (67%) - catches more internet users than XGBoost")