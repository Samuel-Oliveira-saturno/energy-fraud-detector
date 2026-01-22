# ==============================
# Import das bibliotecas 
#===============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

model = joblib.load("modelo_rf.pkl")

# =========================
# Configurações da página
# =========================
st.set_page_config(page_title="Fraud Detection", layout="wide")
# =========================
# Logo + Title
# =========================

# =========================
# Header Image (Banner)
# =========================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

header_path = os.path.join(BASE_DIR, "assets", "header.png")
logo_path = os.path.join(BASE_DIR, "assets", "logo.png")


col_logo, col_title = st.columns([1, 5])

with col_logo:
    st.image("assets/logo.png", width=120)

with col_title:
    st.markdown(
        """
        # Energy Fraud Detection  
      
        """
    )

# =========================
# Carregamento dos dados
# =========================
st.sidebar.header("📂 Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload consumption dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset loaded successfully!")


# =========================
# Carregando o modelo treinado
#==========================

st.sidebar.header("📦 Upload Model")

uploaded_model = st.sidebar.file_uploader(
    "Upload trained model (.pkl)",
    type=["pkl"]
)

if uploaded_model is None:
    st.warning("Please upload a trained model (.pkl).")
    st.stop()

model = joblib.load(uploaded_model)
st.success("Model loaded successfully!")



# =========================
# Engenharia de atributos
# =========================
# =========================
# Feature Engineering
# =========================
consumption_cols = [c for c in df.columns if c.startswith("mes_")]

df["consumo_medio"] = df[consumption_cols].mean(axis=1)
df["consumo_std"] = df[consumption_cols].std(axis=1)
df["consumo_min"] = df[consumption_cols].min(axis=1)
df["consumo_max"] = df[consumption_cols].max(axis=1)
df["consumo_range"] = df["consumo_max"] - df["consumo_min"]

df["coef_variacao"] = df["consumo_std"] / (df["consumo_medio"] + 1e-6)

df["consumo_inicio"] = df[consumption_cols[:3]].mean(axis=1)
df["consumo_fim"] = df[consumption_cols[-3:]].mean(axis=1)
df["razao_fim_inicio"] = df["consumo_fim"] / (df["consumo_inicio"] + 1e-6)

df["maior_queda"] = df[consumption_cols].diff(axis=1).min(axis=1)

features = [
    "consumo_medio",
    "consumo_std",
    "consumo_min",
    "consumo_max",
    "consumo_range",
    "coef_variacao",
    "razao_fim_inicio",
    "maior_queda"
]

X = df[features]

df["fraud_score"] = model.predict_proba(X)[:, 1]



# =========================
# Carregamento do modelo
# =========================
model = joblib.load("modelo_rf.pkl")

df["score_fraude"] = model.predict_proba(X)[:, 1]

# =========================
# Classificação de risco
# =========================
st.sidebar.header("🎛️ Risk Threshold")

high_risk_threshold = st.sidebar.slider(
    "High Risk Threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.70,
    step=0.05
)

medium_risk_threshold = st.sidebar.slider(
    "Medium Risk Threshold",
    min_value=0.20,
    max_value=high_risk_threshold - 0.05,
    value=0.40,
    step=0.05
)




def classify_risk(score):
    if score >= high_risk_threshold:
        return "HIGH"
    elif score >= medium_risk_threshold:
        return "MEDIUM"
    else:
        return "LOW"

df["risk_level"] = df["fraud_score"].apply(classify_risk)


# =========================
# Indicadores gerais
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("🔴 High Risk (%)", f"{(df['risk_level']=='HIGH').mean()*100:.1f}%")
col2.metric("🟡 Medium Risk (%)", f"{(df['risk_level']=='MEDIUM').mean()*100:.1f}%")
col3.metric("🟢 Low Risk (%)", f"{(df['risk_level']=='LOW').mean()*100:.1f}%")


# =========================
# Distribuição dos scores
# =========================
st.subheader("Fraud Risk Score Distribution")

fig, ax = plt.subplots()
sns.histplot(df["fraud_score"], bins=30, kde=True, ax=ax)
ax.set_xlabel("Fraud Risk Score")
ax.set_ylabel("Number of Consumers")
st.pyplot(fig)


# =========================
# Análise individual
# =========================
st.subheader("Analysis by Consumer Unit")

idx = st.selectbox(
    "Select Customer:",
    df.index
)

cliente = df.loc[idx]

st.markdown(f"""
### 🔍 Classification: **{cliente['risk_level']} RISK**
**Score:** `{cliente['score_fraude']:.2f}`
""")

st.subheader("🧠 Risk Explanation")

reasons = []

if cliente["coef_variacao"] > df["coef_variacao"].quantile(0.75):
    reasons.append("High consumption variability")

if cliente["maior_queda"] < df["maior_queda"].quantile(0.25):
    reasons.append("Abrupt drop in consumption")

if cliente["razao_fim_inicio"] < 0.8:
    reasons.append("Significant reduction over time")

if cliente["consumo_medio"] < df["consumo_medio"].median():
    reasons.append("Below average consumption")

if reasons:
    for r in reasons:
        st.markdown(f"- ⚠️ {r}")
else:
    st.markdown("✅ Consumption pattern within expected range")



# Gráfico de consumo
fig2, ax2 = plt.subplots()
ax2.plot(consumption_cols, cliente[consumption_cols], marker="o")
ax2.set_title("Monthly History")
ax2.set_ylabel("Consumption")
ax2.set_xlabel("Months")
plt.xticks(rotation=45)

st.pyplot(fig2)

st.subheader("📄 Download Inspection Report")

def generate_pdf(dataframe):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 50, "Energy Fraud Detection Report")

    y = height - 100
    c.setFont("Helvetica", 10)

    stats = [
        f"Total consumers: {len(dataframe)}",
        f"High risk (%): {(dataframe['risk_level']=='HIGH').mean()*100:.2f}",
        f"Medium risk (%): {(dataframe['risk_level']=='MEDIUM').mean()*100:.2f}",
        f"Low risk (%): {(dataframe['risk_level']=='LOW').mean()*100:.2f}",
        f"Average fraud score: {dataframe['fraud_score'].mean():.3f}"
    ]

    for line in stats:
        c.drawString(50, y, line)
        y -= 20

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


#############################
pdf = generate_pdf(df)

st.download_button(
    label="⬇️ Download PDF Report",
    data=pdf,
    file_name="fraud_detection_report.pdf",
    mime="application/pdf"
)


st.subheader("📋 Inspection Priority List")

inspection_table = df[
    ["fraud_score", "risk_level", "consumo_medio", "coef_variacao", "maior_queda"]
].sort_values(by="fraud_score", ascending=False)

st.dataframe(
    inspection_table.head(20),
    use_container_width=True
)
