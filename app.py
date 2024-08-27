import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and preprocessing objects
model = load_model('deep_learning_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to preprocess input data
def preprocess_input(data):
    data_scaled = scaler.transform(data)
    return data_scaled

# Function to make predictions
def predict_class(input_data):
    input_data_preprocessed = preprocess_input(input_data)
    predictions = model.predict(input_data_preprocessed)
    predicted_classes = np.argmax(predictions, axis=1)
    return label_encoder.inverse_transform(predicted_classes)

# Streamlit UI
st.title('DARWIN Dataset Classification')
st.write('Enter feature values to get predictions.')

# Example input fields (adjust as per your features)
features = {}
for col in scaler.get_feature_names_out():
    features[col] = st.number_input(f'{col}', value=0.0)

if st.button('Predict'):
    input_data = pd.DataFrame([features], columns=scaler.get_feature_names_out())
    prediction = predict_class(input_data)
    st.write(f'Predicted class: {prediction[0]}')
