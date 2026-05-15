from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path("model.pkl")
SCALER_PATH = Path("scaler.pkl")


st.set_page_config(
    page_title="Patient Blood Pressure Prediction",
    page_icon="BP",
    layout="centered",
)

st.markdown(
    """
<style>
.main {
    background-color: #f5f7fa;
}

.stButton > button {
    width: 100%;
    background-color: #2f855a;
    color: white;
    height: 3em;
    border-radius: 8px;
    font-size: 18px;
    border: none;
}

.stButton > button:hover {
    background-color: #276749;
}

h1 {
    text-align: center;
    color: #2c3e50;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


if not MODEL_PATH.exists():
    st.error("model.pkl file not found")
    st.stop()

model = load_pickle(MODEL_PATH)
scaler = load_pickle(SCALER_PATH) if SCALER_PATH.exists() else None

feature_names = list(
    getattr(
        model,
        "feature_names_in_",
        [
            "Patient_Number",
            "Level_of_Hemoglobin",
            "Genetic_Pedigree_Coefficient",
            "Age",
            "BMI",
            "Sex",
            "Pregnancy",
            "Smoking",
            "Physical_activity",
            "salt_content_in_the_diet",
            "alcohol_consumption_per_day",
            "Level_of_Stress",
            "Chronic_kidney_disease",
            "Adrenal_and_thyroid_disorders",
        ],
    )
)


st.title("Patient Blood Pressure Prediction System")
st.write(
    "This machine learning app predicts whether a patient may have abnormal blood pressure "
    "based on health parameters."
)

with st.form("prediction_form"):
    patient_number = st.number_input("Patient Number", min_value=1, max_value=100000, value=1)

    col1, col2 = st.columns(2)
    with col1:
        hemoglobin = st.number_input("Level of Hemoglobin", min_value=0.0, max_value=25.0, value=12.0)
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda value: "Female" if value == 1 else "Male")
        smoking = st.selectbox("Smoking", options=[0, 1], format_func=lambda value: "Yes" if value else "No")
        physical_activity = st.number_input("Physical Activity", min_value=0, max_value=50000, value=25000)
        alcohol = st.number_input("Alcohol Consumption Per Day", min_value=0.0, max_value=500.0, value=0.0)
        chronic_kidney_disease = st.selectbox(
            "Chronic Kidney Disease",
            options=[0, 1],
            format_func=lambda value: "Yes" if value else "No",
        )

    with col2:
        genetic_pedigree = st.number_input(
            "Genetic Pedigree Coefficient",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=30.0)
        pregnancy = st.selectbox("Pregnancy", options=[0, 1], format_func=lambda value: "Yes" if value else "No")
        salt = st.number_input("Salt Content in the Diet", min_value=0, max_value=50000, value=25000)
        stress = st.selectbox("Level of Stress", options=[1, 2, 3])
        adrenal_thyroid = st.selectbox(
            "Adrenal and Thyroid Disorders",
            options=[0, 1],
            format_func=lambda value: "Yes" if value else "No",
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "Patient_Number": patient_number,
                "Level_of_Hemoglobin": hemoglobin,
                "Genetic_Pedigree_Coefficient": genetic_pedigree,
                "Age": age,
                "BMI": bmi,
                "Sex": sex,
                "Pregnancy": pregnancy,
                "Smoking": smoking,
                "Physical_activity": physical_activity,
                "salt_content_in_the_diet": salt,
                "alcohol_consumption_per_day": alcohol,
                "Level_of_Stress": stress,
                "Chronic_kidney_disease": chronic_kidney_disease,
                "Adrenal_and_thyroid_disorders": adrenal_thyroid,
            }
        ],
        columns=feature_names,
    )

    prediction_input = scaler.transform(input_data) if scaler is not None else input_data
    prediction = model.predict(prediction_input)

    confidence = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(prediction_input)
        confidence = np.max(probability) * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Abnormal blood pressure detected")
    else:
        st.success("Normal blood pressure")

    if confidence is not None:
        st.info(f"Confidence Score: {confidence:.2f}%")

st.markdown("---")
st.write("Developed using Python, Machine Learning, and Streamlit")
