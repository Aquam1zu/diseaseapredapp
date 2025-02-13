import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model
import wikipedia
import tensorflow as tf

# Set Page to Wide Mode
st.set_page_config(layout="wide")  

# Custom CSS for better layout and alignment
st.markdown("""
    <style>
        /* Center align the main container */
        .main {
            text-align: center;
        }

        /* Fix button size */
        div.stButton > button {
            width: 250px;
            height: 50px;
            font-size: 18px;
            border-radius: 10px;
            background-color: #FF4B4B;
            color: white;
        }

        /* Adjust multi-select box width */
        div[data-testid="stMultiSelect"] {
            width: 100%;
        }

        /* Align content inside columns */
        section.main > div {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# UI Layout - Using a Better Column Distribution
st.title("🩺 Disease Prediction System")
st.write("### Select symptoms to predict possible diseases.")

# Create columns with a better width ratio
col1, col2 = st.columns([1, 2])  

with col1:
    selected_symptoms = st.multiselect("Select Symptoms:", ["Fever", "Cough", "Fatigue", "Shortness of Breath"])  # Example list

    # Center the button properly
    button_container = st.container()
    with button_container:
        predict_button = st.button("🔍 Predict Disease")

if predict_button:
    with col2:
        st.success("🎯 Predicted Disease: **Influenza**")
        st.write("🟢 Confidence: **82.5%**")

        # Example Top 5 diseases likelihood
        top_5_diseases = {"Influenza": 0.825, "Pneumonia": 0.65, "Bronchitis": 0.48, "COVID-19": 0.39, "Asthma": 0.30}
        st.write("### 📊 Likelihood of Top 5 Diseases:")
        st.bar_chart(pd.DataFrame(top_5_diseases.values(), index=top_5_diseases.keys(), columns=["Likelihood"]))
