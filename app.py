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
    # Print feature names expected by the scaler
    st.write("### Features expected by the scaler:")
    st.write(top_features)

    # Print feature names in the input DataFrame
    st.write("### Features in the uploaded DataFrame:")
    st.write(input_df.columns.tolist())
    
    # Ensure all required features are present
    missing_features = [feat for feat in top_features if feat not in input_df.columns]
    
    # Log missing features if any
    if missing_features:
        st.warning(f"Missing features: {missing_features}")
    
    # Add missing features with default values
    for feat in missing_features:
        input_df[feat] = 0
    
    # Reorder columns to match the training set
    input_df = input_df[top_features]
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    return input_scaled

def main():
    st.title("Deep Learning Model Deployment with Streamlit")
    
    st.write("""
    ### Upload your filtered dataset to get predictions automatically
    """)

    # File uploader to upload CSV
    uploaded_file = st.file_uploader("Upload your input CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        input_df = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write("### Uploaded Data")
        st.write(input_df.head())

        # Preprocess input
        try:
            input_processed = preprocess_input(input_df)

            # Make predictions
            predictions = model.predict(input_processed)
            pred_classes = (predictions > 0.5).astype(int).flatten()  # Convert to binary class
            pred_labels = le.inverse_transform(pred_classes)  # Convert to original labels

            # Create a DataFrame with the predictions
            results_df = input_df.copy()
            results_df['Prediction'] = pred_labels

            # Display results
            st.write("### Predictions")
            st.write(results_df[['Prediction']])

            # Add a download button for predictions
            st.download_button(
                label="Download Predictions",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name='predictions.csv',
                mime='text/csv',
            )
        except ValueError as e:
            st.error(f"Error in preprocessing: {e}")

if __name__ == '__main__':
    main()
