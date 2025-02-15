import os
import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model
import wikipedia
import matplotlib.pyplot as plt

# Google Drive file IDs
CSV_FILE_ID = "1SOGfczIm_XcFJqBxOaOB7kFsBQn3ZSv5"
MODEL_FILE_ID = "1ojNVvOuEb6JyhknTyDVKV6IZrcMTHvog"

# File paths
csv_path = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
model_path = "disease_prediction_model.h5"

def analyze_symptom_significance(model, selected_symptoms, predicted_disease_index, SYMPTOMS):
    """Analyzes the significance of selected symptoms for the predicted disease."""
    weights = model.layers[0].get_weights()[0]
    
    significance_scores = {}
    for symptom in selected_symptoms:
        symptom_index = SYMPTOMS.index(symptom)
        significance = abs(weights[symptom_index][predicted_disease_index])
        significance_scores[symptom] = significance
    
    significance_df = pd.DataFrame.from_dict(
        significance_scores, 
        orient='index', 
        columns=['Significance']
    ).sort_values('Significance', ascending=False)
    
    return significance_df

def plot_symptom_significance(significance_df):
    """Creates a horizontal bar plot of symptom significance."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(significance_df) * 0.4)))
    
    significance_df.plot(
        kind='barh',
        ax=ax,
        color='#FF4B4B',
        alpha=0.6
    )
    
    ax.set_title('Symptom Significance Analysis', pad=20)
    ax.set_xlabel('Relative Significance')
    ax.set_ylabel('Symptoms')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
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

    col1, col2 = st.columns([1, 2])

    with col1:
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
                predicted_disease_index = list(DISEASES).index(predicted_disease)
                confidence_score = top_5_diseases[predicted_disease] * 100

                with col2:
                    st.success(f"🎯 Predicted Disease: **{predicted_disease}**")
                    st.write(f"🟢 Confidence: **{confidence_score:.2f}%**")

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
                    st.write("This shows how much each symptom contributed to the prediction:")
                    
                    significance_df = analyze_symptom_significance(
                        model, 
                        selected_symptoms, 
                        predicted_disease_index,
                        SYMPTOMS
                    )
                    
                    most_sig_symptom = significance_df.index[0]
                    most_sig_value = significance_df.iloc[0]['Significance']
                    st.write(f"**Most significant symptom:** {most_sig_symptom} (Relative importance: {most_sig_value:.4f})")
                    
                    fig = plot_symptom_significance(significance_df)
                    st.pyplot(fig)

                    st.write("### 📋 Detailed Symptom Significance Scores")
                    st.dataframe(significance_df.style.format({'Significance': '{:.4f}'}))

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
