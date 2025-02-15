import os
import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model
import wikipedia
import matplotlib.pyplot as plt

# Set wide layout
st.set_page_config(layout="wide")

# Google Drive file IDs
CSV_FILE_ID = "1SOGfczIm_XcFJqBxOaOB7kFsBQn3ZSv5"
MODEL_FILE_ID = "1ojNVvOuEb6JyhknTyDVKV6IZrcMTHvog"

# File paths
csv_path = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
model_path = "disease_prediction_model.h5"

def analyze_symptom_significance(model, selected_symptoms, prediction_array, SYMPTOMS):
    """
    Analyzes the significance of selected symptoms using prediction differences.
    Uses a different approach that doesn't rely on model weights.
    """
    significance_scores = {}
    baseline_prediction = prediction_array[0]
    
    # Create base symptom array with all selected symptoms
    base_symptoms = np.array([[1 if symptom in selected_symptoms else 0 for symptom in SYMPTOMS]])
    
    # Test each symptom's significance by removing it
    for symptom in selected_symptoms:
        # Create a copy of the base symptoms
        test_symptoms = base_symptoms.copy()
        # Find the index of the current symptom
        symptom_index = SYMPTOMS.index(symptom)
        # Remove this symptom (set to 0)
        test_symptoms[0][symptom_index] = 0
        
        # Get new prediction without this symptom
        new_prediction = model.predict(test_symptoms)[0]
        
        # Calculate significance as the difference in prediction
        significance = np.abs(baseline_prediction - new_prediction).max()
        significance_scores[symptom] = float(significance)
    
    # Create DataFrame and sort by significance
    significance_df = pd.DataFrame.from_dict(
        significance_scores, 
        orient='index', 
        columns=['Significance']
    ).sort_values('Significance', ascending=False)
    
    return significance_df

def plot_symptom_significance(significance_df):
    """Creates a horizontal bar plot of symptom significance."""
    fig, ax = plt.subplots(figsize=(7, min(2.5, len(significance_df) * 0.25)))  # Adjust the height dynamically
    significance_df.plot(
        kind='barh',
        ax=ax,
        color='#FF4B4B',
        alpha=0.6
    )
    
    ax.set_title('Symptom Significance Analysis', pad=15)
    ax.set_xlabel('Impact on Prediction')
    ax.set_ylabel('Symptoms')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()

    # Resize the figure to fit the app's layout
    fig.set_size_inches(9, min(2.5, len(significance_df) * 0.25))  # Ensure it's not too large

    return fig

@st.cache_data
def get_disease_description(disease_name):
    try:
        page = wikipedia.page(disease_name)
        return page.summary
    except Exception as e:
        return f"Could not fetch description for {disease_name}."

@st.cache_resource
def load_data_and_model():
    """Load and cache the dataset and model."""
    if not os.path.exists(csv_path):
        with st.spinner('Downloading dataset...'):
            try:
                gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", csv_path, quiet=False)
            except Exception as e:
                st.error(f"Error downloading dataset: {str(e)}")
                return None, None, None, None

    if not os.path.exists(model_path):
        with st.spinner('Downloading model...'):
            try:
                gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", model_path, quiet=False)
            except Exception as e:
                st.error(f"Error downloading model: {str(e)}")
                return None, None, None, None

    try:
        df = pd.read_csv(csv_path)
        symptoms = [col for col in df.columns if col.lower() != "diseases"]
        diseases = df["diseases"].unique()
        model = load_model(model_path)
        return df, symptoms, diseases, model
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def main():
    st.title("🩺 Disease Prediction System")
    
    # Load data and model
    df, SYMPTOMS, DISEASES, model = load_data_and_model()
    
    if df is None or model is None:
        st.error("Error: Could not load necessary files. Please check the logs above.")
        return

    st.write("### Select symptoms to predict possible diseases.")
    selected_symptoms = st.multiselect("Select Symptoms:", SYMPTOMS)
    predict_button = st.button("🔍 Predict Disease")

    if predict_button and selected_symptoms:
        with st.spinner('Analyzing symptoms...'):
            try:
                symptom_values = np.array([[1 if symptom in selected_symptoms else 0 for symptom in SYMPTOMS]])
                prediction = model.predict(symptom_values)
                
                top_5_indices = np.argsort(prediction[0])[-5:][::-1]
                top_5_diseases = {DISEASES[i]: prediction[0][i] for i in top_5_indices}
                
                predicted_disease = list(top_5_diseases.keys())[0]
                confidence_score = top_5_diseases[predicted_disease] * 100
    
                st.success(f"🎯 Predicted Disease: **{predicted_disease}**")
                st.write(f"🟢 Confidence: **{confidence_score:.2f}%**")
                
                # Recommendation for low confidence score
                if confidence_score < 70:
                    st.warning("⚠️ The confidence in this prediction is below 70%. It is highly recommended that you consult a doctor for a more accurate diagnosis and better treatment options.")
                
                description = get_disease_description(predicted_disease)
                st.write(f"### ℹ️ About {predicted_disease}:")
                st.write(description)
                
                st.write("### 📊 Likelihood of Top 5 Diseases:")
                st.bar_chart(pd.DataFrame(
                    top_5_diseases.values(),
                    index=top_5_diseases.keys(),
                    columns=["Likelihood"]
                ))
                
                st.write("### 🔍 Symptom Significance Analysis")
                significance_df = analyze_symptom_significance(
                    model, 
                    selected_symptoms, 
                    prediction,
                    SYMPTOMS
                )
                
                fig = plot_symptom_significance(significance_df)
                st.pyplot(fig)
        
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
