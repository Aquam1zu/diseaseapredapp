import os
import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model
import wikipedia
import matplotlib.pyplot as plt

# [Previous imports and setup code remains the same...]

def analyze_symptom_significance(model, selected_symptoms, predicted_disease_index):
    """
    Analyzes the significance of selected symptoms for the predicted disease.
    
    Args:
        model: The trained Keras model
        selected_symptoms: List of symptoms selected by the user
        predicted_disease_index: Index of the predicted disease in the model's output
        
    Returns:
        DataFrame containing symptom significance scores
    """
    # Get the weights from the first dense layer
    weights = model.layers[0].get_weights()[0]
    
    # Calculate significance scores for selected symptoms
    significance_scores = {}
    for symptom in selected_symptoms:
        symptom_index = SYMPTOMS.index(symptom)
        # Get the weight connecting this symptom to the predicted disease
        significance = abs(weights[symptom_index][predicted_disease_index])
        significance_scores[symptom] = significance
    
    # Create DataFrame and sort by significance
    significance_df = pd.DataFrame.from_dict(
        significance_scores, 
        orient='index', 
        columns=['Significance']
    ).sort_values('Significance', ascending=True)
    
    return significance_df

def plot_symptom_significance(significance_df):
    """
    Creates a horizontal bar plot of symptom significance.
    
    Args:
        significance_df: DataFrame containing symptom significance scores
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(significance_df) * 0.4)))
    
    # Create horizontal bar plot
    significance_df.plot(
        kind='barh',
        ax=ax,
        color='#FF4B4B',
        alpha=0.6
    )
    
    # Customize plot
    ax.set_title('Symptom Significance Analysis', pad=20)
    ax.set_xlabel('Relative Significance')
    ax.set_ylabel('Symptoms')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    st.title("🩺 Disease Prediction System")
    st.write("### Select symptoms to predict possible diseases.")

    # Create three columns for better layout
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        selected_symptoms = st.multiselect("Select Symptoms:", SYMPTOMS)
        predict_button = st.button("🔍 Predict Disease")

    if predict_button and selected_symptoms:
        symptom_values = np.array([[1 if symptom in selected_symptoms else 0 for symptom in SYMPTOMS]])
        prediction = model.predict(symptom_values)

        # Get top 5 predicted diseases
        top_5_indices = np.argsort(prediction[0])[-5:][::-1]
        top_5_diseases = {DISEASES[i]: prediction[0][i] for i in top_5_indices}

        # Get the most likely disease
        predicted_disease = list(top_5_diseases.keys())[0]
        predicted_disease_index = list(DISEASES).index(predicted_disease)
        confidence_score = top_5_diseases[predicted_disease] * 100

        with col2:
            st.success(f"🎯 Predicted Disease: **{predicted_disease}**")
            st.write(f"🟢 Confidence: **{confidence_score:.2f}%**")

            # Fetch and display disease description
            description = get_disease_description(predicted_disease)
            st.write(f"### ℹ️ About {predicted_disease}:")
            st.write(description)

        with col3:
            # Display bar chart for top 5 diseases
            st.write("### 📊 Likelihood of Top 5 Diseases:")
            st.bar_chart(pd.DataFrame(top_5_diseases.values(), index=top_5_diseases.keys(), columns=["Likelihood"]))

        # Analyze and display symptom significance
        st.write("### 🔍 Symptom Significance Analysis")
        st.write("This chart shows how much each symptom contributed to the prediction:")
        
        significance_df = analyze_symptom_significance(model, selected_symptoms, predicted_disease_index)
        fig = plot_symptom_significance(significance_df)
        st.pyplot(fig)

        # Display detailed significance scores in a table
        st.write("### 📋 Detailed Symptom Significance Scores")
        st.dataframe(significance_df.style.format({'Significance': '{:.4f}'}))
