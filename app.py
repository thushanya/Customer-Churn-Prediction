import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Churn AI Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #ffffff;
}

.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
models = {
    "Logistic Regression": pickle.load(open("models/lr.pkl", "rb")),
    "Random Forest": pickle.load(open("models/rf.pkl", "rb")),
    "XGBoost": pickle.load(open("models/xgb.pkl", "rb"))
}

scaler = pickle.load(open("models/scaler.pkl", "rb"))
columns = pickle.load(open("models/columns.pkl", "rb"))

# ---------------- HEADER ----------------
st.title("📊 Telecom Customer Churn Intelligence Dashboard")
st.write("AI-powered prediction system for customer retention strategy")

st.divider()

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Customer Input Panel")

model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_name]

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 100.0)

# ---------------- PREDICTION ----------------
if st.sidebar.button("🚀 Predict Churn"):

    input_df = pd.DataFrame(0, index=[0], columns=columns)

    # numeric features
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly
    input_df["TotalCharges"] = total

    # categorical encoding (must match training dummies)
    input_df[f"gender_{gender}"] = 1
    input_df["SeniorCitizen"] = senior
    input_df[f"Partner_{partner}"] = 1
    input_df[f"Dependents_{dependents}"] = 1
    input_df[f"InternetService_{internet}"] = 1
    input_df[f"Contract_{contract}"] = 1

    # align columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # scale
    input_scaled = scaler.transform(input_df)

    # prediction
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # ---------------- RESULTS ----------------
    st.subheader("📌 Prediction Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", model_name)

    with col2:
        st.metric("Churn Probability", f"{prob:.2f}")

    with col3:
        risk = "HIGH 🔴" if prob > 0.6 else "MEDIUM 🟠" if prob > 0.3 else "LOW 🟢"
        st.metric("Risk Level", risk)

    st.divider()

    if prediction == 1:
        st.error("⚠ Customer is LIKELY TO CHURN")
    else:
        st.success("✅ Customer is LIKELY TO STAY")

    # ---------------- BUSINESS LOGIC ----------------
    st.subheader("💡 Business Recommendation")

    if prob > 0.6:
        st.warning("👉 Offer discount / retention call immediately")
    elif prob > 0.3:
        st.info("👉 Monitor customer behavior closely")
    else:
        st.success("👉 No immediate action required")

# ---------------- ANALYTICS DASHBOARD ----------------
st.divider()
st.header("📈 Analytics Dashboard")

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Distribution")

    fig, ax = plt.subplots()
    df["Churn"].value_counts().plot(
        kind="bar",
        ax=ax,
        color=["green", "red"]
    )
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.subheader("Monthly Charges vs Churn")

    fig, ax = plt.subplots()
    df.boxplot(column="MonthlyCharges", by="Churn", ax=ax)
    ax.set_title("")
    st.pyplot(fig)

# ---------------- INSIGHTS ----------------
st.subheader("📌 Key Business Insights")

st.info("""
✔ Month-to-month contracts → highest churn  
✔ High monthly charges → higher churn risk  
✔ Longer tenure → lower churn risk  
✔ Fiber optic users → more likely to churn  
""")