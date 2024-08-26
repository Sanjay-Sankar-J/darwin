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

def preprocess_input(input_df):
    # Handle missing values if necessary
    # Here we assume input_df has all necessary columns
    
    # Encode categorical variables if any
    # This step depends on how you handle categorical features
    # For simplicity, assume all features are numerical
    
    # Scale numerical features
    input_scaled = scaler.transform(input_df[top_features])
    
    return input_scaled

def main():
    st.title("Deep Learning Model Deployment with Streamlit")
    
    st.write("""
    ### Upload a CSV file with the following features to get predictions
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        # Ensure the required features are present
        missing_features = [feat for feat in top_features if feat not in input_df.columns]
        if missing_features:
            st.error(f"The following required features are missing: {missing_features}")
        else:
            # Preprocess input
            input_processed = preprocess_input(input_df)
            
            # Make predictions
            prediction = model.predict(input_processed)
            pred_class = (prediction > 0.5).astype(int).flatten()
            pred_label = le.inverse_transform(pred_class)
            
            # Display results
            results = pd.DataFrame({
                'Prediction': pred_label
            })
            st.write("### Predictions")
            st.write(results)
            
            # Optionally, allow downloading the results
            st.download_button(
                label="Download Predictions",
                data=results.to_csv(index=False).encode('utf-8'),
                file_name='predictions.csv',
                mime='text/csv',
            )

if __name__ == '__main__':
    main()
