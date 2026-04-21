import streamlit as st
import joblib, pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Uganda Internet Access Predictor",
    page_icon="🌐",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0f2027 100%);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.03em;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    margin-top: 0.8rem;
}

/* ── Section cards ── */
.section-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.2s;
}
.section-card:hover {
    border-color: rgba(56,189,248,0.2);
}
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #1e293b !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:focus {
    border-color: rgba(56,189,248,0.4) !important;
}
label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.02em !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 2rem;
    font-family: 'Sora', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(14,165,233,0.35);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Result box ── */
.result-box {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-yes {
    background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(16,185,129,0.06));
    border: 1px solid rgba(52,211,153,0.3);
}
.result-no {
    background: linear-gradient(135deg, rgba(248,113,113,0.12), rgba(239,68,68,0.06));
    border: 1px solid rgba(248,113,113,0.3);
}
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-yes .result-label { color: #34d399; }
.result-no  .result-label { color: #f87171; }
.result-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: #94a3b8;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1, #34d399) !important;
    border-radius: 100px !important;
}
.stProgress > div > div {
    background: #1e293b !important;
    border-radius: 100px !important;
    height: 10px !important;
}

/* ── Info sidebar ── */
.info-pill {
    background: #1e293b;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.83rem;
    color: #cbd5e1;
    line-height: 1.6;
}
.info-pill strong { color: #38bdf8; }

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.82rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: #64748b; }
.stat-val { color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-weight: 600; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Hide streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
model     = joblib.load("nn_model.pkl")
scaler    = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")

SCALER_COLS = ['age', 'Grade', 'energysource', 'waterdrinking', 'toilet',
               'rururb_1.0', 'rururb_2.0', 'sex_1', 'sex_2',
               'Region15_1', 'Region15_2', 'Region15_3',
               'literacy_encoded', 'attending_encoded']

MODEL_COLS = ['age', 'Grade', 'phone', 'computer', 'television', 'radio',
              'energysource', 'waterdrinking', 'toilet', 'bank_account',
              'rururb_1.0', 'rururb_2.0', 'sex_1', 'sex_2',
              'Region15_1', 'Region15_2', 'Region15_3',
              'literacy_encoded', 'attending_encoded']

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🌐 Uganda Internet Access Predictor</div>
    <div class="hero-sub">Predict household internet access using socioeconomic & demographic indicators</div>
    <div class="hero-badge">MLPClassifier · 2014 Uganda National Census · 3.5M Records</div>
</div>
""", unsafe_allow_html=True)

# ── Layout: form (left wide) + info (right narrow) ────────────────────────────
form_col, info_col = st.columns([2.6, 1], gap="large")

with form_col:
    c1, c2 = st.columns(2, gap="medium")

    # ── Demographics ──────────────────────────────────────────────────────────
    with c1:
        st.markdown('<div class="section-card"><div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
        age    = st.number_input("Age (years)", min_value=0, max_value=120, value=25)
        sex    = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x==1 else "Female")
        region = st.selectbox("Region", [1, 2, 3],
                     format_func=lambda x: {1:"Central", 2:"Eastern / Northern", 3:"Western"}[x])
        rururb = st.selectbox("Area Type", [1, 2], format_func=lambda x: "🏙️ Urban" if x==1 else "🌾 Rural")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">🎓 Education</div>', unsafe_allow_html=True)
        grade     = st.number_input("Highest Grade Completed", min_value=0, max_value=99, value=7)
        literacy  = st.selectbox("Literacy Level", [1, 2, 3, 4],
                       format_func=lambda x: {1:"Reads & Writes", 2:"Reads Only",
                                              3:"Cannot Read/Write", 4:"N/A (Child)"}[x])
        attending = st.selectbox("Currently in School", [0, 1],
                       format_func=lambda x: "No" if x==0 else "Yes")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Assets & Infrastructure ───────────────────────────────────────────────
    with c2:
        st.markdown('<div class="section-card"><div class="section-title">📱 Asset Ownership</div>', unsafe_allow_html=True)
        phone        = st.selectbox("Mobile Phone",  [0, 1], format_func=lambda x: "❌ No" if x==0 else "✅ Yes")
        computer     = st.selectbox("Computer",      [0, 1], format_func=lambda x: "❌ No" if x==0 else "✅ Yes")
        television   = st.selectbox("Television",    [0, 1], format_func=lambda x: "❌ No" if x==0 else "✅ Yes")
        radio        = st.selectbox("Radio",         [0, 1], format_func=lambda x: "❌ No" if x==0 else "✅ Yes")
        bank_account = st.selectbox("Bank Account",  [0, 1], format_func=lambda x: "❌ No" if x==0 else "✅ Yes")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">🏠 Infrastructure</div>', unsafe_allow_html=True)
        energysource  = st.selectbox("Energy Source", list(range(1, 10)),
                           format_func=lambda x: {1:"Grid Electricity", 2:"Solar", 3:"Generator",
                                                  4:"Paraffin/Kerosene", 5:"Firewood", 6:"Charcoal",
                                                  7:"Biogas", 8:"Other", 9:"None"}.get(x, str(x)))
        waterdrinking = st.selectbox("Drinking Water Source", list(range(1, 9)),
                           format_func=lambda x: {1:"Piped into Dwelling", 2:"Piped (Yard/Plot)",
                                                  3:"Public Tap", 4:"Borehole/Well", 5:"Protected Spring",
                                                  6:"Unprotected Source", 7:"Rain Water", 8:"Other"}.get(x, str(x)))
        toilet        = st.selectbox("Toilet Facility", list(range(1, 7)),
                           format_func=lambda x: {1:"Flush Toilet", 2:"VIP Latrine", 3:"Pit Latrine (covered)",
                                                  4:"Pit Latrine (open)", 5:"No Facility", 6:"Other"}.get(x, str(x)))
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────────────────────────
    predict = st.button("🔍  Predict Internet Access", use_container_width=True)

    # ── Result ────────────────────────────────────────────────────────────────
    if predict:
        rururb_1          = 1 if rururb == 1 else 0
        rururb_2          = 1 if rururb == 2 else 0
        sex_1             = 1 if sex == 1 else 0
        sex_2             = 1 if sex == 2 else 0
        region_1          = 1 if region == 1 else 0
        region_2          = 1 if region == 2 else 0
        region_3          = 1 if region == 3 else 0
        literacy_encoded  = literacy - 1
        attending_encoded = attending

        scaler_input = pd.DataFrame([[
            age, grade, energysource, waterdrinking, toilet,
            rururb_1, rururb_2, sex_1, sex_2,
            region_1, region_2, region_3,
            literacy_encoded, attending_encoded
        ]], columns=SCALER_COLS)

        scaled = scaler.transform(scaler_input)
        s = dict(zip(SCALER_COLS, scaled[0]))

        full_input = pd.DataFrame([[
            s['age'], s['Grade'],
            phone, computer, television, radio,
            s['energysource'], s['waterdrinking'], s['toilet'],
            bank_account,
            s['rururb_1.0'], s['rururb_2.0'],
            s['sex_1'], s['sex_2'],
            s['Region15_1'], s['Region15_2'], s['Region15_3'],
            s['literacy_encoded'], s['attending_encoded']
        ]], columns=MODEL_COLS)

        proba = model.predict_proba(full_input)[0][1]
        pred  = int(proba >= threshold)

        if pred == 1:
            st.markdown(f"""
            <div class="result-box result-yes">
                <div class="result-emoji">✅</div>
                <div class="result-label">Has Internet Access</div>
                <div class="result-conf">Model confidence: {proba:.1%}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-no">
                <div class="result-emoji">❌</div>
                <div class="result-label">No Internet Access</div>
                <div class="result-conf">Model confidence: {1-proba:.1%}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(proba))
        st.caption(f"Raw internet-access probability: **{proba:.1%}** · Decision threshold: {threshold:.2f}")

# ── Right info panel ──────────────────────────────────────────────────────────
with info_col:
    st.markdown("""
    <div class="info-pill">
        <strong>🧠 Model</strong><br>
        Neural Network (MLPClassifier)<br>
        Hidden layers: 100 → 50 neurons<br>
        Activation: ReLU · Scaled inputs
    </div>
    <div class="info-pill">
        <strong>📊 Performance</strong>
        <div class="stat-row"><span class="stat-key">Accuracy</span><span class="stat-val">87%</span></div>
        <div class="stat-row"><span class="stat-key">ROC-AUC</span><span class="stat-val">0.94</span></div>
        <div class="stat-row"><span class="stat-key">F1-Score</span><span class="stat-val">0.87</span></div>
        <div class="stat-row"><span class="stat-key">Training rows</span><span class="stat-val">3.5M</span></div>
    </div>
    <div class="info-pill">
        <strong>📈 Top Predictors</strong>
        <div class="stat-row"><span class="stat-key">Urban/Rural</span><span class="stat-val">40%</span></div>
        <div class="stat-row"><span class="stat-key">Mobile Phone</span><span class="stat-val">24%</span></div>
        <div class="stat-row"><span class="stat-key">Computer</span><span class="stat-val">16%</span></div>
        <div class="stat-row"><span class="stat-key">Education</span><span class="stat-val">12%</span></div>
        <div class="stat-row"><span class="stat-key">Literacy</span><span class="stat-val">8%</span></div>
    </div>
    <div class="info-pill" style="font-size:0.75rem; color:#475569;">
        📌 Based on the 2014 Uganda National Population & Housing Census.<br><br>
        For research and policy planning purposes only.
    </div>
    """, unsafe_allow_html=True)
    