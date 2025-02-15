import os
import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model
import wikipedia

# Initialize session state for file downloads
if 'files_downloaded' not in st.session_state:
    st.session_state.files_downloaded = False

# Google Drive file IDs
CSV_FILE_ID = "1SOGfczIm_XcFJqBxOaOB7kFsBQn3ZSv5"
MODEL_FILE_ID = "1ojNVvOuEb6JyhknTyDVKV6IZrcMTHvog"

# File paths
csv_path = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
model_path = "disease_prediction_model.h5"

# Download and load files
@st.cache_resource
def load_data_and_model():
    # Download CSV
    if not os.path.exists(csv_path):
        with st.spinner('Downloading dataset...'):
            try:
                gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", csv_path, quiet=False)
            except Exception as e:
                st.error(f"Error downloading dataset: {str(e)}")
                return None, None, None, None

    # Download Model
    if not os.path.exists(model_path):
        with st.spinner('Downloading model...'):
            try:
                gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", model_path, quiet=False)
            except Exception as e:
                st.error(f"Error downloading model: {str(e)}")
                return None, None, None, None

    try:
        # Load dataset
        df = pd.read_csv(csv_path)
        symptoms = [col for col in df.columns if col.lower() != "diseases"]
        diseases = df["diseases"].unique()
        
        # Load model
        model = load_model(model_path)
        
        return df, symptoms, diseases, model
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

# Function to get disease description
@st.cache_data
def get_disease_description(disease_name):
    try:
        page = wikipedia.page(disease_name)
        return page.summary
    except Exception as e:
        return f"Could not fetch description for {disease_name}."

# Load data and model
df, SYMPTOMS, DISEASES, model = load_data_and_model()

# Main app
def main():
    st.title("🩺 Disease Prediction System")
    
    if df is None or model is None:
        st.error("Error: Could not load necessary files. Please check the logs above.")
        return

    st.write("### Select symptoms to predict possible diseases.")

    # Create three columns for better layout
    col1, col2 = st.columns([1, 2])

    with col1:
        selected_symptoms = st.multiselect("Select Symptoms:", SYMPTOMS)
        predict_button = st.button("🔍 Predict Disease")

    if predict_button and selected_symptoms:
        with st.spinner('Analyzing symptoms...'):
            try:
                # Prepare input
                symptom_values = np.array([[1 if symptom in selected_symptoms else 0 for symptom in SYMPTOMS]])
                
                # Make prediction
                prediction = model.predict(symptom_values)
                
                # Get top 5 predicted diseases
                top_5_indices = np.argsort(prediction[0])[-5:][::-1]
                top_5_diseases = {DISEASES[i]: prediction[0][i] for i in top_5_indices}
                
                # Get the most likely disease
                predicted_disease = list(top_5_diseases.keys())[0]
                confidence_score = top_5_diseases[predicted_disease] * 100

                with col2:
                    st.success(f"🎯 Predicted Disease: **{predicted_disease}**")
                    st.write(f"🟢 Confidence: **{confidence_score:.2f}%**")

                    # Fetch and display disease description
                    description = get_disease_description(predicted_disease)
                    st.write(f"### ℹ️ About {predicted_disease}:")
                    st.write(description)

                    # Display bar chart for top 5 diseases
                    st.write("### 📊 Likelihood of Top 5 Diseases:")
                    st.bar_chart(pd.DataFrame(
                        top_5_diseases.values(),
                        index=top_5_diseases.keys(),
                        columns=["Likelihood"]
                    ))

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
