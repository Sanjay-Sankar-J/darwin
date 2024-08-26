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

def preprocess_input(input_dict):
    # Convert input_dict to a DataFrame
    input_df = pd.DataFrame([input_dict], columns=top_features)
    
    # Ensure all required features are present
    for feat in top_features:
        if feat not in input_df.columns:
            input_df[feat] = 0  # Add missing features with default values
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    return input_scaled

def main():
    st.title("Deep Learning Model Deployment with Streamlit")
    
    st.write("""
    ### Enter values for the following features to get predictions
    """)
    
    # Create a form for user inputs
    with st.form(key='input_form'):
        inputs = {}
        for feature in top_features:
            # Use appropriate input widgets based on feature types
            inputs[feature] = st.number_input(f"Enter value for {feature}", value=0.0, format="%.2f")
        
        # Submit button
        submit_button = st.form_submit_button(label="Get Prediction")
        
        if submit_button:
            # Preprocess input
            input_processed = preprocess_input(inputs)
            
            if input_processed is not None:
                # Make predictions
                prediction = model.predict(input_processed)
                pred_class = (prediction > 0.5).astype(int).flatten()  # Convert to binary class
                pred_label = le.inverse_transform(pred_class)  # Convert to original labels
                
                # Display results
                st.write("### Prediction")
                st.write(pred_label[0])
                
                # Optionally, you can add a download button for individual predictions
                st.download_button(
                    label="Download Prediction",
                    data=pd.DataFrame({'Prediction': [pred_label[0]]}).to_csv(index=False).encode('utf-8'),
                    file_name='prediction.csv',
                    mime='text/csv',
                )

if __name__ == '__main__':
    main()
