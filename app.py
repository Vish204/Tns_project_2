import streamlit as st
import requests

st.title("Heart Disease Prediction")

# Input fields (raw categorical values)
# age = st.number_input('Age', min_value=0, max_value=120, value=63, format="%.2f")
# sex = st.radio('Sex', options=[0, 1], index=1, format_func=lambda x: 'Female' if x == 0 else 'Male')
# resting_blood_pressure = st.number_input('Resting Blood Pressure', min_value=0, max_value=250, value=145, format="%.2f")
# cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=233, format="%.2f")
# fasting_blood_sugar = st.radio('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
# max_heart_rate = st.number_input('Max Heart Rate Achieved', min_value=0, max_value=250, value=150, format="%.2f")
# exercise_induced_angina = st.radio('Exercise Induced Angina', options=[0, 1])
# st_depression = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=2.3, format="%.2f")
# num_major_vessels = st.selectbox('Number of major vessels colored by fluoroscopy', options=[0, 1, 2, 3])
# chest_pain_type = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
# resting_ecg = st.selectbox('Resting ECG', options=[0, 1, 2])
# st_slope = st.selectbox('Slope of the peak exercise', options=[0, 1, 2])
# thalassemia = st.selectbox('Thalassemia', options=[0, 1, 2, 3])

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
    data = {
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

    response = requests.post('http://127.0.0.1:8000/predict', json=data)

    if response.status_code == 200:
        result = response.json()
        pred = result.get('prediction')
        prob = result.get('probability_of_disease')
        st.success(f"Prediction: {'Heart Disease Present' if pred == 1 else 'No Heart Disease'}")
        st.info(f"Prediction Probability: {prob*100:.2f}%")
    else:
        st.error(f"Error: {response.text}")














# import streamlit as st
# import requests

# st.title("Heart Disease Prediction")

# # Input fields for the features
# age = st.number_input('Age', min_value=0, max_value=120, value=63)
# sex = st.radio('Sex', options=[0, 1], index=1, format_func=lambda x: 'Female' if x == 0 else 'Male')
# chestpaintype = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
# restingbloodpressure = st.number_input('Resting Blood Pressure', min_value=0, max_value=250, value=145)
# cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=233)
# fastingbloodsugar = st.radio('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
# restingecg = st.selectbox('Resting ECG', options=[0, 1, 2])
# maxheartrate = st.number_input('Max Heart Rate Achieved', min_value=0, max_value=250, value=150)
# exerciseinducedangina = st.radio('Exercise Induced Angina', options=[0, 1])
# stdepression = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=2.3)
# slope = st.selectbox('Slope of the peak exercise', options=[0, 1, 2])
# majorvessels = st.selectbox('Number of major vessels colored by fluoroscopy', options=[0, 1, 2, 3])
# thalassemia = st.selectbox('Thalassemia', options=[0, 1, 2, 3])

# if st.button('Predict'):
#     # Prepare data payload
#     # After collecting inputs:
#     data = {
#         "age": age,
#         "sex": sex,
#         "chest_pain_type": chestpaintype,
#         "resting_blood_pressure": restingbloodpressure,
#         "cholesterol": cholesterol,
#         "fasting_blood_sugar": fastingbloodsugar,
#         "resting_ecg": restingecg,
#         "max_heart_rate": maxheartrate,
#         "exercise_induced_angina": exerciseinducedangina,
#         "st_depression": stdepression,
#         "st_slope": slope,
#         "num_major_vessels": majorvessels,
#         "thalassemia": thalassemia
#     }

    

#     # Call the FastAPI predict endpoint
#     response = requests.post('http://127.0.0.1:8000/predict', json=data)

#     if response.status_code == 200:
#         result = response.json()
#         pred = result.get('prediction')
#         prob = result.get('probability_of_disease')
#         st.success(f"Prediction: {'Heart Disease Present' if pred == 1 else 'No Heart Disease'}")
#         st.info(f"Prediction Probability: {prob*100:.2f}%")
#     else:
#         st.error(f"Error: {response.text}")
