import os
import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model
import wikipedia
import tensorflow as tf

# Force TensorFlow to use the CPU (if you don't need GPU acceleration)
tf.config.set_visible_devices([], 'GPU')

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

# Symptom selection
selected_symptoms = st.multiselect("Select symptoms:", SYMPTOMS)

if st.button("Predict Disease"):
    # Prepare input data for prediction
    symptom_values = np.array([[1 if symptom in selected_symptoms else 0 for symptom in SYMPTOMS]])
    
    # Predict using the model
    prediction = model.predict(symptom_values)
    
    # Get all prediction probabilities
    prediction_probs = prediction[0]
    
    # Get predicted disease (max probability)
    predicted_index = np.argmax(prediction_probs)
    predicted_disease = DISEASES[predicted_index]
    
    # Get the confidence score (percentage)
    confidence_score = prediction_probs[predicted_index]
    confidence_percentage = round(confidence_score * 100, 2)
    
    # Display the predicted disease and confidence
    st.success(f"Predicted Disease: **{predicted_disease}**")
    st.write(f"Confidence: **{confidence_percentage}%**")
    
    # Fetch and display disease description from Wikipedia
    description = get_disease_description(predicted_disease)
    st.write(f"**About {predicted_disease}:** {description}")
    
    # Display bar chart for the top 5 diseases likelihood
    disease_confidence = {DISEASES[i]: prediction_probs[i] for i in range(len(DISEASES))}
    
    # Sort by confidence in descending order and get top 5
    top_5_diseases = dict(sorted(disease_confidence.items(), key=lambda item: item[1], reverse=True)[:5])
    
    # Create a bar chart of the top 5 diseases
    st.bar_chart(pd.DataFrame(top_5_diseases.values(), index=top_5_diseases.keys(), columns=["Likelihood"]))
