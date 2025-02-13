import os
import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model
import wikipedia

# Set Page to Wide Mode
st.set_page_config(layout="wide")

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

# Load the model
model = load_model(model_path)

# Function to get disease description
def get_disease_description(disease_name):
    try:
        page = wikipedia.page(disease_name)
        return page.summary
    except wikipedia.exceptions.DisambiguationError:
        return f"Multiple diseases found for {disease_name}, please check the exact name."
    except wikipedia.exceptions.HTTPTimeoutError:
        return "Error: Could not fetch data from Wikipedia."
    except Exception as e:
        return f"Error: {str(e)}"

# Custom CSS
st.markdown("""
    <style>
        .main { text-align: center; }
        div.stButton > button {
            width: 250px;
            height: 50px;
            font-size: 18px;
            border-radius: 10px;
            background-color: #FF4B4B;
            color: white;
        }
        div[data-testid="stMultiSelect"] { width: 100%; }
        section.main > div { display: flex; justify-content: center; }
    </style>
""", unsafe_allow_html=True)

# UI Layout with Two Columns
st.title("🩺 Disease Prediction System")
st.write("### Select symptoms to predict possible diseases.")

col1, col2 = st.columns([1, 2])  # Left column for input, right column for results

with col1:
    selected_symptoms = st.multiselect("Select Symptoms:", SYMPTOMS)
    predict_button = st.button("🔍 Predict Disease")

if predict_button:
    symptom_values = np.array([[1 if symptom in selected_symptoms else 0 for symptom in SYMPTOMS]])
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
        st.bar_chart(pd.DataFrame(top_5_diseases.values(), index=top_5_diseases.keys(), columns=["Likelihood"]))
