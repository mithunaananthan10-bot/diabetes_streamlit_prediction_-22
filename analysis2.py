import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import streamlit as st

# page config (must use st.)
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# ---------- BLUE & WHITE THEME ----------
st.markdown("""
<style>
body {
    background-color: #f4f8ff;
}
h1, h2, h3 {
    color: #0b3c7d;
    text-align: center;
}
.stButton>button {
    background-color: #0b5ed7;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    background-color: #084298;
}
</style>
""", unsafe_allow_html=True)

# read dataset
df = pd.read_csv("diabetes.csv")

# preprocessing (same order as yours)
cols = ["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"]
for col in cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].mean())

# split X and y
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# train test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model1 = RandomForestClassifier(n_estimators=50, random_state=42)
model1.fit(x_train, y_train)

# prediction on test data
pred = model1.predict(x_test)

# accuracy
acc = accuracy_score(y_test, pred)

# ================= STREAMLIT UI (bottom like yours) =================

st.markdown("<h1>🩺 Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown(f"<h3>Model Accuracy: {acc*100:.2f}%</h3>", unsafe_allow_html=True)

st.subheader("Enter patient details:")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Enter number of pregnancies", 0, 20, 0)
    glucose = st.number_input("Enter glucose level", 0, 300, 120)
    bp = st.number_input("Enter blood pressure", 0, 200, 70)
    skin = st.number_input("Enter skin thickness", 0, 100, 20)

with col2:
    bmi = st.number_input("Enter BMI", 0.0, 70.0, 25.0)
    insulin = st.number_input("Enter insulin level", 0, 900, 80)
    dpf = st.number_input("Enter diabetes pedigree function", 0.0, 3.0, 0.5)
    age = st.number_input("Enter age", 1, 120, 30)
    
if st.button("Predict"):
    user_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=X.columns
    )

    result = model1.predict(user_data)[0]
    prob = model1.predict_proba(user_data)[0][1]

    if result == 1:
        st.error(f"⚠️ Person is likely Diabetic (Probability: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Person is NOT Diabetic (Probability: {prob*100:.2f}%)")