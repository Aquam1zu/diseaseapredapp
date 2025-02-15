import os
import streamlit as st
import pandas as pd
import numpy as np
import gdown
from tensorflow.keras.models import load_model
import wikipedia
import matplotlib.pyplot as plt

# [Previous imports and cache setup remain the same...]

def analyze_symptom_significance(model, selected_symptoms, predicted_disease_index, SYMPTOMS):
    """Analyzes the significance of selected symptoms for the predicted disease."""
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
    ).sort_values('Significance', ascending=False)  # Changed to descending order
    
    return significance_df

def plot_symptom_significance(significance_df):
    """Creates a horizontal bar plot of symptom significance."""
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

# [Previous data loading functions remain the same...]

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
                predicted_disease_index = list(DISEASES).index(predicted_disease)
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

                    # Analyze and display symptom significance
                    st.write("### 🔍 Symptom Significance Analysis")
                    st.write("This shows how much each symptom contributed to the prediction:")
                    
                    significance_df = analyze_symptom_significance(
                        model, 
                        selected_symptoms, 
                        predicted_disease_index,
                        SYMPTOMS
                    )
                    
                    # Display most significant symptom
                    most_sig_symptom = significance_df.index[0]
                    most_sig_value = significance_df.iloc[0]['Significance']
                    st.write(f"**Most significant symptom:** {most_sig_symptom} (Relative importance: {most_sig_value:.4f})")
                    
                    # Plot significance
                    fig = plot_symptom_significance(significance_df)
                    st.pyplot(fig)

                    # Display detailed significance scores
                    st.write("### 📋 Detailed Symptom Significance Scores")
                    st.dataframe(significance_df.style.format({'Significance': '{:.4f}'}))

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
