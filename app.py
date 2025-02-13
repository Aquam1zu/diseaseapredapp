import os
import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model
import wikipedia

# Google Drive file IDs
CSV_FILE_ID = "1SOGfczIm_XcFJqBxOaOB7kFsBQn3ZSv5"
MODEL_FILE_ID = "1ojNVvOuEb6JyhknTyDVKV6IZrcMTHvog"

@st.cache_data
def load_data():
    # Download Dataset
    csv_path = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    if not os.path.exists(csv_path):
        gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", csv_path, quiet=False)
    
    # Load dataset
    df = pd.read_csv(csv_path)
    SYMPTOMS = [col for col in df.columns if col.lower() != "diseases"]
    DISEASES = df["diseases"].unique()
    return df, SYMPTOMS, DISEASES

@st.cache_resource
def load_model_from_drive():
    # Download Model
    model_path = "disease_prediction_model.h5"
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", model_path, quiet=False)
    
    # Load the model from .h5 file
    model = load_model(model_path)
    return model

# Function to get disease description
def get_disease_description(disease_name):
    try:
        # Search for disease page on Wikipedia
        page = wikipedia.page(disease_name)
        return page.summary  # Return the first paragraph (summary)
    except wikipedia.exceptions.DisambiguationError as e:
        # In case of disambiguation (multiple pages with similar names)
        return f"Multiple diseases found for {disease_name}, please check the exact name."
    except wikipedia.exceptions.HTTPTimeoutError:
        return "Error: Could not fetch data from Wikipedia."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("Disease Prediction System")
st.write("Select symptoms to predict the possible disease.")

# Load data and model
df, SYMPTOMS, DISEASES = load_data()
model = load_model_from_drive()

# Symptom selection
selected_symptoms = st.multiselect("Select symptoms:", SYMPTOMS)

if st.button("Predict Disease"):
    # Show the spinner while the prediction is being processed
    with st.spinner('Predicting disease...'):
        symptom_values = np.array([[1 if symptom in selected_symptoms else 0 for symptom in SYMPTOMS]])
        prediction = model.predict(symptom_values)
    
    predicted_index = np.argmax(prediction)
    predicted_disease = DISEASES[predicted_index]
    
    confidence_score = prediction[0][predicted_index]  # Extract confidence score
    confidence_percentage = round(confidence_score * 100, 2)  # Convert to percentage

    st.success(f"Predicted Disease: **{predicted_disease}**")
    st.write(f"Confidence: **{confidence_percentage}%**")
    
    # Fetch and display disease description from Wikipedia
    description = get_disease_description(predicted_disease)
    st.write(f"**About {predicted_disease}:** {description}")
