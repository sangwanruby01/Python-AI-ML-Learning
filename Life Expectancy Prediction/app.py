import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title="Life Expectancy Dashboard", layout="wide")

st.title("🌍 Life Expectancy Prediction Dashboard")
st.markdown("Professional ML Dashboard using Streamlit")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("Life Expectancy Data.csv")
df.columns = df.columns.str.strip()
df = df.dropna()

st.sidebar.success("Dashboard Loaded ✔")

# ----------------------------
# TABS (PROFESSIONAL STYLE)
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data",
    "📈 Visualization",
    "🤖 Model",
    "🎯 Prediction"
])

# ----------------------------
# TAB 1 - DATA
# ----------------------------
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

# ----------------------------
# TAB 2 - VISUALIZATION
# ----------------------------
with tab2:
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution")

    num_col = st.selectbox("Select column", df.select_dtypes(include=np.number).columns)

    fig2, ax2 = plt.subplots()
    sns.histplot(df[num_col], kde=True, ax=ax2)
    st.pyplot(fig2)

# ----------------------------
# TAB 3 - MODEL
# ----------------------------
with tab3:
    st.subheader("Model Training")

    target = "Life expectancy"

    X = df.select_dtypes(include=np.number).drop(columns=[target], errors="ignore")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2:.2f}")

    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred)
    ax3.set_xlabel("Actual")
    ax3.set_ylabel("Predicted")
    ax3.set_title("Actual vs Predicted")
    st.pyplot(fig3)

# ----------------------------
# TAB 4 - PREDICTION
# ----------------------------
with tab4:
    st.subheader("Make Prediction")

    target = "Life expectancy"
    X = df.select_dtypes(include=np.number).drop(columns=[target], errors="ignore")

    input_data = []

    for col in X.columns:
        val = st.number_input(f"{col}", float(df[col].mean()))
        input_data.append(val)

    if st.button("Predict"):
        model = LinearRegression()
        model.fit(X, df[target])

        prediction = model.predict(np.array(input_data).reshape(1, -1))

        st.success(f"🌟 Predicted Life Expectancy: {prediction[0]:.2f}")