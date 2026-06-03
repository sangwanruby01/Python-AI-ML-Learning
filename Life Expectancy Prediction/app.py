import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# LOAD MODEL
# =========================
model = joblib.load("life_expectancy_model.pkl")

st.set_page_config(page_title="Life Expectancy AI", layout="wide")

# =========================
# TITLE
# =========================
st.title("🌍 Life Expectancy Prediction Dashboard")
st.markdown("Predict life expectancy using WHO health & economic indicators")

# =========================
# LAYOUT
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", 2000, 2015, 2015)
    status = st.selectbox("Status", ["Developing", "Developed"])
    status = 1 if status == "Developed" else 0
    adult_mortality = st.number_input("Adult Mortality")

with col2:
    infant_deaths = st.number_input("Infant Deaths")
    alcohol = st.number_input("Alcohol")
    bmi = st.number_input("BMI")
    hiv = st.number_input("HIV/AIDS")
    gdp = st.number_input("GDP")

with col3:
    schooling = st.number_input("Schooling")
    percentage_expenditure = st.number_input("Percentage Expenditure")
    hepatitis_b = st.number_input("Hepatitis B")
    measles = st.number_input("Measles")
    population = st.number_input("Population")

# extra inputs (hidden section)
under_five_deaths = st.number_input("Under Five Deaths")
polio = st.number_input("Polio")
total_expenditure = st.number_input("Total Expenditure")
diphtheria = st.number_input("Diphtheria")
thinness_1_19 = st.number_input("Thinness 1-19 Years")
thinness_5_9 = st.number_input("Thinness 5-9 Years")
income = st.number_input("Income Composition")

# =========================
# INPUT ARRAY
# =========================
input_data = np.array([[
    year,
    status,
    adult_mortality,
    infant_deaths,
    alcohol,
    percentage_expenditure,
    hepatitis_b,
    measles,
    bmi,
    under_five_deaths,
    polio,
    total_expenditure,
    diphtheria,
    hiv,
    gdp,
    population,
    thinness_1_19,
    thinness_5_9,
    income,
    schooling
]])

# =========================
# PREDICTION
# =========================
st.markdown("---")

if st.button("🔮 Predict Life Expectancy"):
    prediction = model.predict(input_data)[0]

    st.success(f"🌟 Predicted Life Expectancy: {prediction:.2f} years")

    # Insight box
    if prediction > 75:
        st.info("High life expectancy country")
    elif prediction > 60:
        st.warning("Medium life expectancy country")
    else:
        st.error("Low life expectancy country")

# =========================
# FEATURE IMPORTANCE CHART
# =========================
st.markdown("---")
st.subheader("📊 Feature Importance (Model Insight)")

features = [
    "Year","Status","Adult Mortality","Infant Deaths","Alcohol",
    "Percentage Expenditure","Hepatitis B","Measles","BMI",
    "Under Five Deaths","Polio","Total Expenditure","Diphtheria",
    "HIV/AIDS","GDP","Population","Thinness 1-19","Thinness 5-9",
    "Income","Schooling"
]

importance = model.feature_importances_

df_imp = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(df_imp.set_index("Feature"))

# =========================
# FOOTER INSIGHT
# =========================
st.markdown("---")
st.info("Built using Random Forest Regressor | WHO Life Expectancy Dataset")