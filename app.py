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

# Define the expected features
required_features = [
    "total_time23", "total_time15", "air_time15", "air_time23", "air_time17",
    "pressure_mean5", "total_time17", "total_time3", "total_time22", "total_time6",
    "air_time22", "total_time7", "paper_time23", "total_time9", "total_time8",
    "paper_time17", "disp_index23", "paper_time20", "air_time5", "total_time11",
    "mean_speed_in_air17", "paper_time8", "air_time6", "paper_time9", "mean_jerk_in_air17",
    "air_time24", "disp_index17", "disp_index8", "mean_gmrt7", "pressure_var19",
    "mean_gmrt8", "num_of_pendown19", "gmrt_in_air17", "mean_acc_in_air17", "paper_time22",
    "mean_speed_on_paper10", "pressure_mean4", "gmrt_on_paper8", "air_time7", "mean_jerk_in_air3",
    "pressure_mean14", "gmrt_on_paper17", "paper_time3", "air_time16", "mean_jerk_on_paper5",
    "num_of_pendown9", "total_time16", "total_time12", "mean_jerk_on_paper24", "paper_time10"
]

def preprocess_input(input_df):
    # Ensure all required features are present and add missing features with default values
    for feat in required_features:
        if feat not in input_df.columns:
            input_df[feat] = 0  # Adding missing features with default values
    
    # Reorder columns to match the required feature list
    input_df = input_df[required_features]
    
    # Print the ordered DataFrame for debugging
    st.write("### Ordered DataFrame for Scaler:")
    st.write(input_df.head())

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

        # Ensure the input data contains the required features
        missing_features = [feat for feat in required_features if feat not in input_df.columns]
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
        else:
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
