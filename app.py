import os
import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model
import wikipedia
import tensorflow as tf

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

# Page Layout Styling
st.set_page_config(layout="wide")  # Expands the app width

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            text-align: center;
        }
        div.stButton > button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            font-size: 18px;
            border-radius: 10px;
        }
        div[data-testid="stVerticalBlock"] {  /* Adjust general container */
            align-items: center;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

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

# Function to fetch disease description
def get_disease_description(disease_name):
    try:
        page = wikipedia.page(disease_name)
        return page.summary  
    except wikipedia.exceptions.DisambiguationError:
        return f"Multiple results for {disease_name}, refine your query."
    except wikipedia.exceptions.HTTPTimeoutError:
        return "Error fetching data from Wikipedia."
    except Exception as e:
        return f"Error: {str(e)}"

# === UI Layout ===
st.title("🩺 Disease Prediction System")

st.write("### Select symptoms to predict possible diseases.")

# Organizing UI elements using columns
col1, col2 = st.columns([2, 3])  # Left for selection, right for output

with col1:
    selected_symptoms = st.multiselect("Select Symptoms:", SYMPTOMS)
    if st.button("🔍 Predict Disease"):
        symptom_values = np.array([[1 if symptom in selected_symptoms else 0 for symptom in SYMPTOMS]])
        prediction = model.predict(symptom_values)
        prediction_probs = prediction[0]
        
        # Get the top predicted disease
        predicted_index = np.argmax(prediction_probs)
        predicted_disease = DISEASES[predicted_index]
        confidence_score = round(prediction_probs[predicted_index] * 100, 2)

        # Show results on the right
        with col2:
            st.success(f"### 🎯 Predicted Disease: **{predicted_disease}**")
            st.write(f"🟢 Confidence: **{confidence_score}%**")
            description = get_disease_description(predicted_disease)
            st.write(f"📝 **About {predicted_disease}:** {description}")

            # Show Top 5 Predictions as Bar Chart
            disease_confidence = {DISEASES[i]: prediction_probs[i] for i in range(len(DISEASES))}
            top_5_diseases = dict(sorted(disease_confidence.items(), key=lambda item: item[1], reverse=True)[:5])

            st.write("### 📊 Likelihood of Top 5 Diseases:")
            st.bar_chart(pd.DataFrame(top_5_diseases.values(), index=top_5_diseases.keys(), columns=["Likelihood"]))
