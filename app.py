import os
import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model

# Google Drive file IDs
CSV_FILE_ID = "1SOGfczIm_XcFJqBxOaOB7kFsBQn3ZSv5"
MODEL_FILE_ID = "1ojNVvOuEb6JyhknTyDVKV6IZrcMTHvog"

# Download Dataset
csv_path = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
if not os.path.exists(csv_path):
    gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", csv_path, quiet=False)

# Load dataset
df = pd.read_csv(csv_path)
SYMPTOMS = [col for col in df.columns if col.lower() != "diseases"]
DISEASES = df["diseases"].unique()

# Download Model
model_path = "disease_prediction_model.h5"
if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", model_path, quiet=False)

# Load the model from .h5 file
model = load_model(model_path)

# Streamlit UI
st.title("Disease Prediction System")
st.write("Select symptoms to predict the possible disease.")

# Symptom selection
selected_symptoms = st.multiselect("Select symptoms:", SYMPTOMS)

if st.button("Predict Disease"):
    symptom_values = np.array([[1 if symptom in selected_symptoms else 0 for symptom in SYMPTOMS]])
    prediction = model.predict(symptom_values)
    predicted_index = np.argmax(prediction)
    predicted_disease = DISEASES[predicted_index]

    st.success(f"Predicted Disease: **{predicted_disease}**")
