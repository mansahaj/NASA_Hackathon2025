import os
import pickle

import streamlit as st

import pandas as pd
import numpy as np

import joblib

# --- Global Variables --- #
model_options = {
    "Random Forest": "../nasa-apps-backend/models/random_forest_model.pkl",
}

# --- PAGE CONFIGURATION --- #
st.set_page_config(
    page_title="Hunting for Exoplanets with AI", 
    page_icon="🤖",
    layout="centered"
)
def config():
    st.markdown("""
        <style>
            /* Make main container moderately wider */
            .block-container {
                max-width: 1100px;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            p {
                text-align: justify;
            }
        </style>
    """, unsafe_allow_html=True)


# --- HELPER FUNCTIONS FOR LOADING MODELS/SCALERS --- #
def load_model_and_scaler(model):
    # Load Model
    model = joblib.load(model)

    # Load Scaler
    scaler = joblib.load('../nasa-apps-backend/scalers/random_forest_scaler.pkl')

    return (model, scaler)

"""Exoplanet's (Challenge) Explained"""
def intro():
        # Title
    st.title("Hunting for Exoplanets with AI")
    
    # Horizontal Line
    st.markdown("---")

    # Explain Explanets
    st.markdown("""
    Welcome to this project!  
    This application demonstrates the functionality of our model, which was developed to analyze and interpret data efficiently. 
    We aim to provide insights and visualizations that make complex results easy to understand.  

    The system integrates machine learning techniques and an intuitive interface to create an accessible and interactive experience for users. 
    """)

    # Transit Method Video
    video_url = "https://www.youtube.com/watch?v=bv2BV82J0Jk"
    st.markdown(f"""
        <div style="display: flex; justify-content: center;">
            <iframe width="640" height="360" 
                    src="{video_url.replace('watch?v=', 'embed/')}" 
                    frameborder="0" allowfullscreen>
            </iframe>
        </div>
    """, unsafe_allow_html=True)

    # Explain Transit Method
    st.markdown("""
    Below is a deeper explanation of the methods and goals behind this project.  
    We focused on optimizing performance and ensuring the system remains interpretable and transparent to its users.  

    Future iterations will include real-time updates, advanced analytics, and user-driven customization options.
    """)

    # Horizontal Line
    st.markdown("---")

"""Running the model"""
def run_model():

    selected_model = st.selectbox("Select a Model:", list(model_options.keys()))

    try:
        model_path = model_options[selected_model]
        (md, sc) = load_model_and_scaler(model_path)
        st.success(f"{selected_model} loaded successfully!")
    except Exception as e:
        st.error(f"Error loading {selected_model}: {e}")

    # CSV upload section
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            candidates_df = pd.read_csv(uploaded_file)
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            
            st.markdown("### Preview of Uploaded Data")
            st.dataframe(candidates_df.head())

            if st.button("Run Model"):
                try:
                    # Load candidate data
                    candidates_df = pd.read_csv("../nasa-apps-backend/Data/candidate_rows.csv")

                    # Scale the features using the random forest scaler
                    candidates_scaled = sc.transform(candidates_df)

                    # Predict using random forest model
                    pred = md.predict(candidates_scaled)

                    # Combine results
                    results = candidates_df.copy()
                    results["Prediction"] = pred

                    # Display output
                    st.markdown("### Prediction Results")
                    st.dataframe(results.head())

                except FileNotFoundError:
                    st.error("'candidate_rows.csv' file not found. Please check the file path.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

def display_results():
    pass

def main():
    config()

    intro()

    run_model()

    display_results()



if __name__ == "__main__":
    main()