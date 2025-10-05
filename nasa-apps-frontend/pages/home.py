import os
import pickle

import streamlit as st

import pandas as pd
import numpy as np

import joblib


"""PAGE CONFIGURATION"""
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

"""HELPER FUNCTIONS FOR LOADING MODELS/SCALERS"""
def load_models_and_scaler():
    # Load Models
    md_random_forest = joblib.load('../nasa-apps-backend/models/random_forest_model.pkl')

    # Load Scalers
    sc_random_forest = joblib.load('../nasa-apps-backend/scalers/random_forest_scaler.pkl')

    return (md_random_forest, sc_random_forest)

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

"""Demo using candidates data"""
def demo():
    if st.button("Run Demo"):
        try:
            # Load candidate data
            candidates_df = pd.read_csv("../nasa-apps-backend/Data/candidate_rows.csv")

            # Scale the features using the random forest scaler
            candidates_scaled = sc_rd.transform(candidates_df)

            # Predict using random forest model
            md_rd_pred = md_rd.predict(candidates_scaled)

            # Combine results
            results = candidates_df.copy()
            results["Prediction"] = md_rd_pred

            # Display output
            st.markdown("### Prediction Results")
            st.dataframe(results.head())

        except FileNotFoundError:
            st.error("'candidate_rows.csv' file not found. Please check the file path.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


"""Grab all models and scalers"""
(md_rd, sc_rd) = load_models_and_scaler()

def main():
    config()

    intro()

    demo()



if __name__ == "__main__":
    main()