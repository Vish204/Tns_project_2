#for independent running of app.py
import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# Load model, scaler, feature columns (same as backend)
with open("Heart_disease_rf_model.pkl", "rb") as f_model:
    model = pickle.load(f_model)

with open("scaler.pkl", "rb") as f_scaler:
    scaler = pickle.load(f_scaler)

with open("feature_columns.json", "r") as f_feat:
    feature_columns = json.load(f_feat)

st.title("Heart Disease Prediction")

# User inputs
age = st.number_input('Age', min_value=0.0, max_value=120.0, value=63.0, format="%.2f")
sex = st.radio('Sex', options=[0, 1], index=1, format_func=lambda x: 'Female' if x == 0 else 'Male')
resting_blood_pressure = st.number_input('Resting Blood Pressure', min_value=0.0, max_value=250.0, value=145.0, format="%.2f")
cholesterol = st.number_input('Cholesterol', min_value=0.0, max_value=600.0, value=233.0, format="%.2f")
fasting_blood_sugar = st.radio('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
max_heart_rate = st.number_input('Max Heart Rate Achieved', min_value=0.0, max_value=250.0, value=150.0, format="%.2f")
exercise_induced_angina = st.radio('Exercise Induced Angina', options=[0, 1])
st_depression = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=2.3, format="%.2f")
num_major_vessels = st.selectbox('Number of major vessels colored by fluoroscopy', options=[0, 1, 2, 3])
chest_pain_type = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
resting_ecg = st.selectbox('Resting ECG', options=[0, 1, 2])
st_slope = st.selectbox('Slope of the peak exercise', options=[0, 1, 2])
thalassemia = st.selectbox('Thalassemia', options=[0, 1, 2, 3])


if st.button('Predict'):
    try:
        # Create input dictionary with raw inputs like FastAPI expects
        input_dict = {
            "age": age,
            "sex": sex,
            "resting_blood_pressure": resting_blood_pressure,
            "cholesterol": cholesterol,
            "fasting_blood_sugar": fasting_blood_sugar,
            "max_heart_rate": max_heart_rate,
            "exercise_induced_angina": exercise_induced_angina,
            "st_depression": st_depression,
            "num_major_vessels": num_major_vessels,
            "chest_pain_type": chest_pain_type,
            "resting_ecg": resting_ecg,
            "st_slope": st_slope,
            "thalassemia": thalassemia
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # One-hot encode categorical columns as in backend
        cat_cols = ['chest_pain_type', 'resting_ecg', 'st_slope', 'thalassemia']
        input_df = pd.get_dummies(input_df, columns=cat_cols)

        # Add missing columns with zeros
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match model training
        input_df = input_df[feature_columns]

        # Scale numeric columns (match backend)
        numeric_cols = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate', 'st_depression']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success("Prediction: Heart Disease Present")
        else:
            st.success("Prediction: No Heart Disease")

        st.info(f"Prediction Probability: {prediction_proba * 100:.2f}%")

    except Exception as e:
        st.error(f"Prediction error: {e}")












#for dependent of app.py and main.py
#docker using below code





# import streamlit as st
# import requests

# st.title("Heart Disease Prediction")



# age = st.number_input('Age', min_value=0.0, max_value=120.0, value=63.0, format="%.2f")
# sex = st.radio('Sex', options=[0, 1], index=1, format_func=lambda x: 'Female' if x == 0 else 'Male')
# resting_blood_pressure = st.number_input('Resting Blood Pressure', min_value=0.0, max_value=250.0, value=145.0, format="%.2f")
# cholesterol = st.number_input('Cholesterol', min_value=0.0, max_value=600.0, value=233.0, format="%.2f")
# fasting_blood_sugar = st.radio('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
# max_heart_rate = st.number_input('Max Heart Rate Achieved', min_value=0.0, max_value=250.0, value=150.0, format="%.2f")
# exercise_induced_angina = st.radio('Exercise Induced Angina', options=[0, 1])
# st_depression = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=2.3, format="%.2f")
# num_major_vessels = st.selectbox('Number of major vessels colored by fluoroscopy', options=[0, 1, 2, 3])
# chest_pain_type = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
# resting_ecg = st.selectbox('Resting ECG', options=[0, 1, 2])
# st_slope = st.selectbox('Slope of the peak exercise', options=[0, 1, 2])
# thalassemia = st.selectbox('Thalassemia', options=[0, 1, 2, 3])


# if st.button('Predict'):
#     data = {
#         "age": age,
#         "sex": sex,
#         "resting_blood_pressure": resting_blood_pressure,
#         "cholesterol": cholesterol,
#         "fasting_blood_sugar": fasting_blood_sugar,
#         "max_heart_rate": max_heart_rate,
#         "exercise_induced_angina": exercise_induced_angina,
#         "st_depression": st_depression,
#         "num_major_vessels": num_major_vessels,
#         "chest_pain_type": chest_pain_type,
#         "resting_ecg": resting_ecg,
#         "st_slope": st_slope,
#         "thalassemia": thalassemia
#     }

#     response = requests.post('http://127.0.0.1:8000/predict', json=data)

#     if response.status_code == 200:
#         result = response.json()
#         pred = result.get('prediction')
#         prob = result.get('probability_of_disease')
#         st.success(f"Prediction: {'Heart Disease Present' if pred == 1 else 'No Heart Disease'}")
#         st.info(f"Prediction Probability: {prob*100:.2f}%")
#     else:
#         st.error(f"Error: {response.text}")


