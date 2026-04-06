
import streamlit as st
import joblib, pandas as pd

model     = joblib.load("lightgbm_internet_model.pkl")
features  = joblib.load("features.pkl")
threshold = joblib.load("threshold.pkl")

st.title("🌐 Internet Access Predictor — Uganda Census")

age          = st.number_input("Age", min_value=0, max_value=120, value=25)
sex          = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x==1 else "Female")
grade        = st.number_input("Education Grade", min_value=0, max_value=99, value=7)
literacy     = st.selectbox("Literacy", [1, 2, 3, 4],
                   format_func=lambda x: {1:"Reads & Writes", 2:"Reads Only",
                                          3:"Cannot Read", 4:"N/A"}[x])
phone        = st.selectbox("Owns Phone",      [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
computer     = st.selectbox("Owns Computer",   [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
radio        = st.selectbox("Owns Radio",      [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
television   = st.selectbox("Owns Television", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
meals        = st.selectbox("Meals per day", [1, 2, 3, 4, 5])
livelihood   = st.number_input("Livelihood source code", min_value=0, value=10)
rururb       = st.selectbox("Area type", [1, 2], format_func=lambda x: "Urban" if x==1 else "Rural")
energysource = st.number_input("Energy source code", min_value=0, value=1)
bank_account = st.selectbox("Has Bank Account", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

input_data = pd.DataFrame([[
    age, sex, grade, literacy, phone, computer,
    radio, television, meals, livelihood, rururb, energysource, bank_account
]], columns=features)

if st.button("Predict"):
    proba = model.predict_proba(input_data)[0][1]
    pred  = int(proba >= threshold)
    label = "✅ Has Internet" if pred == 1 else "❌ No Internet"
    st.subheader(f"Prediction: {label}")
    st.progress(float(proba))
    st.write(f"Confidence: {proba:.1%}")