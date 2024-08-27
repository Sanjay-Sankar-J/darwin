# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the scaler, feature selector, and label encoder
scaler = joblib.load('scaler.pkl')
top_features = joblib.load('top_features.pkl')
le = joblib.load('label_encoder.pkl')

# Load the trained model
model = load_model('deep_learning_model.h5')

# Function to preprocess user input
def preprocess_input(input_data):
    # Convert input_data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Handle missing values if necessary
    # (Assuming user inputs all required fields)
    
    # Encode categorical variables
    # For simplicity, assume categorical variables are already numerical or handled

    # Scale numerical features
    input_df_scaled = scaler.transform(input_df)
    
    # Select top features
    input_selected = input_df[top_features]
    input_scaled = scaler.transform(input_selected)
    
    return input_scaled

# Define the input fields
def user_input_features():
    # Since there are 50 features, it's impractical to have all as input fields.
    # Instead, you might allow uploading a file or selecting from predefined options.
    # For demonstration, let's assume top_features are known and manually add a few.
    
    # Example with dummy feature names. Replace with actual feature names.
    input_data = {}
    for feature in top_features:
        input_data[feature] = st.number_input(f'Input {feature}', value=0.0)
    return input_data

def main():
    st.title("Deep Learning Model Deployment with Streamlit")
    
    st.write("""
    ### Enter the feature values to get a prediction
    """)
    
    input_data = user_input_features()
    
    if st.button('Predict'):
        # Preprocess the input
        input_processed = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(input_processed)
        
        # For binary classification
        pred_class = (prediction > 0.5).astype(int)[0][0]
        pred_label = le.inverse_transform([pred_class])[0]
        
        # For multiclass classification
        # pred_class = np.argmax(prediction, axis=1)
        # pred_label = le.inverse_transform(pred_class)[0]
        
        st.write(f"### Predicted Class: {pred_label}")

if __name__ == '__main__':
    main()
