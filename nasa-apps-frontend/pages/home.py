import os
import pickle

import streamlit as st

import json
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib

from type_exoplanet import classify_exoplanet_type

# --- Global Variables --- #
model_options = {
    "Random Forest": "../nasa-apps-backend/models/random_forest_model.pkl",
    "SVM": "../nasa-apps-backend/models/SVM.pkl",
    "MLP": "../nasa-apps-backend/models/MLP.pkl",
    "CNN": "../nasa-apps-backend/models/exoplanet_cnn_model50epochs.h5"
}

# --- PAGE CONFIGURATION --- #
st.set_page_config(
    page_title="Hunting for Exoplanets with AI", 
    page_icon="🪐",
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

# --- Exoplanet's (Challenge) Explained --- #
def intro():
    # Title
    st.title("Hunting for Exoplanets with AI")
    
    # Horizontal Line
    st.markdown("---")

    # Explain Explanets
    st.markdown("""
    **Exoplanets** are planets that orbit stars beyond our Solar System. 
    Scientists have discovered thousands of these distant worlds, ranging from gas giants larger than Jupiter 
    to small, rocky planets similar to Earth. Each new discovery helps researchers better understand how planetary systems form, 
    evolve, and perhaps even support life beyond our own.

    To learn more about exoplanets and NASA’s ongoing missions to discover them, visit  
    [**NASA’s Exoplanet Exploration Program**](https://exoplanets.nasa.gov/)
    """)

    # Horizontal Divider
    st.markdown("---")

    # Transit Method Explanation
    st.markdown("""
    One of the primary ways astronomers detect exoplanets is through the **Transit Method**.  
    When a planet passes in front of its host star, it causes a small, temporary dip in the star’s brightness. By carefully analyzing these periodic dips using space telescopes like *Kepler* and *TESS*, researchers can determine the planet’s size, orbital period, and sometimes even atmospheric properties.

    The video below provides a simple visualization of how the transit method works:
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

    # Horizontal Line
    st.markdown("---")

# --- Running the model --- #
def run_model():
    st.markdown("## Run Model")

    # --- Load Manifest ---
    try:
        with open("../nasa-apps-backend/metadata/manifest.json", "r") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        st.error("manifest.json not found in models directory.")
        manifest = {"models": []}

    # --- Model Selection ---
    selected_model = st.selectbox("Select a Model:", list(model_options.keys()))

    # --- Load Model & Scaler ---
    try:
        model_path = model_options[selected_model]
        (md, sc) = load_model_and_scaler(model_path)
        st.success(f"{selected_model} loaded successfully!")
    except Exception as e:
        st.error(f"Error loading {selected_model}: {e}")
        return

    # --- Display Model Metrics ---
    model_info = next(
        (m for m in manifest["models"] if m["name"].lower() == selected_model.lower().replace(" ", "")),
        None
    )

    if model_info:
        if model_info.get("status") == "pending":
            st.warning("This model is still being trained and evaluated.")
        else:
            st.markdown("### Model Performance Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{model_info['accuracy']:.3f}")
            col2.metric("Precision", f"{model_info['precision']:.3f}")
            col3.metric("Recall", f"{model_info['recall']:.3f}")
            col4.metric("F1 Score", f"{model_info['f1_score']:.3f}")
            col5.metric("ROC AUC", f"{model_info['roc_auc']:.3f}")

            # --- Confusion Matrix ---
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:  # centers the heatmap
                st.markdown("#### Confusion Matrix")
                cm = np.array(model_info["confusion_matrix"]["matrix"])
                fig, ax = plt.subplots(figsize=(2.5, 2.5))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=["Pred: False", "Pred: True"],
                    yticklabels=["Actual: False", "Actual: True"],
                    annot_kws={"size": 8}
                )
                ax.tick_params(axis='both', labelsize=8)
                ax.set_xlabel("")
                ax.set_ylabel("")
                st.pyplot(fig, use_container_width=False)
    else:
        st.warning("No metric data found for this model in manifest.json.")

    st.markdown("---")

    # --- CSV Upload & Prediction ---
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            candidates_df = pd.read_csv(uploaded_file)
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            
            st.markdown("### Preview of Uploaded Data")
            st.dataframe(candidates_df.head())

            if st.button("Run Model"):
                candidates_scaled = sc.transform(candidates_df)
                preds = md.predict(candidates_scaled)

                # Combine results
                results = candidates_df.copy()
                results["Prediction"] = preds.astype(int)  # only Prediction as int

                # Move Prediction column to the front
                cols = ["Prediction"] + [c for c in results.columns if c != "Prediction"]
                results = results[cols]

                # Save for visualizations
                st.session_state["results"] = results

                # --- Highlight 1s in green ---
                def highlight_predictions(val):
                    return "background-color: lightgreen" if val == 1 else ""

                # Use Styler but only target the 'Prediction' column
                styler = results.style.applymap(highlight_predictions, subset=["Prediction"])

                st.markdown("### Prediction Results")
                st.dataframe(styler)

                # Download button
                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

                # Wrap all visualizations inside an expander
                with st.expander("View Prediction Visualizations and Metrics", expanded=False):

                    # Check if predictions exist in session state
                    if "results" not in st.session_state:
                        st.info("Run a model first to view the results.")
                        return

                    results = st.session_state["results"]

                    if "Prediction" not in results.columns:
                        st.warning("No prediction data found in uploaded results.")
                        return

                    # Label mapping (1 = confirmed, 0 = false positive)
                    label_map = {1: "Confirmed Exoplanet", 0: "False Positive"}
                    results["Label"] = results["Prediction"].map(label_map)

                    counts = results["Label"].value_counts()

                    # --- Summary Stats ---
                    st.markdown("### Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Candidates", len(results))
                    col2.metric("Confirmed Exoplanets", counts.get("Confirmed Exoplanet", 0))
                    col3.metric("False Positives", counts.get("False Positive", 0))

                    st.markdown("---")

                    # --- Charts ---
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.bar(counts.index, counts.values, color=sns.color_palette("viridis", len(counts)))
                    ax.set_ylabel("Count")
                    ax.set_xticklabels(counts.index, rotation=20)
                    st.pyplot(fig, use_container_width=True)

                    fig2, ax2 = plt.subplots(figsize=(4, 4))
                    ax2.pie(
                        counts.values,
                        labels=counts.index,
                        autopct="%1.1f%%",
                        startangle=90,
                        textprops={"fontsize": 9}
                    )
                    ax2.axis("equal")
                    st.pyplot(fig2, use_container_width=True)

                # --- Filter only confirmed exoplanets ---
                confirmed_df = results[results["Prediction"] == 1].copy()
                if confirmed_df.empty:
                    st.warning("No confirmed exoplanets found in this dataset.")
                else:
                    # --- Save CSV of confirmed only ---
                    st.markdown("### Preview of Confirmed Exoplanets")
                    st.dataframe(confirmed_df.head())
                    csv_confirmed = confirmed_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Confirmed Exoplanets CSV",
                        data=csv_confirmed,
                        file_name="confirmed_exoplanets.csv",
                        mime="text/csv"
                    )
                    
                    # Save for visualizations
                    st.session_state["confirmed"] = confirmed_df

                    # Wrap all visualizations inside an expander
                    with st.expander("View Prediction Visualizations and Metrics", expanded=False):
                        # --- Exoplanet Type Metrics (Confirmed Only) ---
                        if "confirmed" in st.session_state:
                            confirmed_df = st.session_state["confirmed"].copy()
                            if not confirmed_df.empty:
                                # Apply classification function
                                confirmed_df["Exoplanet_Type"] = confirmed_df.apply(classify_exoplanet_type, axis=1)

                                st.markdown("### Confirmed Exoplanet Types")
                                type_counts = confirmed_df["Exoplanet_Type"].value_counts()

                                # --- Horizontal Bar Chart ---
                                fig3, ax3 = plt.subplots(figsize=(6, 4))
                                ax3.barh(
                                    type_counts.index,
                                    type_counts.values,
                                    color=sns.color_palette("viridis", len(type_counts)),
                                    height=0.6,
                                    edgecolor='black'
                                )

                                # Add count labels
                                for i, v in enumerate(type_counts.values):
                                    ax3.text(v + max(type_counts.values)*0.01, i, str(v), va='center', fontsize=8)

                                ax3.set_xlabel("Count", fontsize=9)
                                ax3.set_ylabel("Exoplanet Type", fontsize=9)
                                ax3.tick_params(axis='x', labelsize=8)
                                ax3.tick_params(axis='y', labelsize=8)
                                ax3.spines['top'].set_visible(False)
                                ax3.spines['right'].set_visible(False)

                                fig3.tight_layout()
                                st.pyplot(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")



def main():
    config()

    intro()

    run_model()

if __name__ == "__main__":
    main()