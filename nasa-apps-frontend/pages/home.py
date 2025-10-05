import os
import io
import base64

import pickle
import json
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
import streamlit as st

from type_exoplanet import classify_exoplanet_type
logo_path = "logo.png"


# --- Global Variables --- #
model_options = {
    "Random Forest": "../nasa-apps-backend/models/random_forest_model.pkl",
    "SVM": "../nasa-apps-backend/models/SVM.pkl",
    "MLP": "../nasa-apps-backend/models/MLP.pkl",
    "CNN": "../nasa-apps-backend/models/exoplanet_cnn_model50epochs.h5"
}

SCALER_PATHS = {
    "Random Forest": "../nasa-apps-backend/scalers/random_forest_scaler.pkl",
    "SVM": "../nasa-apps-backend/scalers/random_forest_scaler.pkl",  # update if different
    "MLP": "../nasa-apps-backend/scalers/random_forest_scaler.pkl",
    "CNN": None  # CNN might not need scaler
}

# --- PAGE CONFIGURATION --- #
st.set_page_config(
    page_title="Hunting for Exoplanets with AI", 
    page_icon="🪐",
    layout="wide"  # better for sidebar + main content
)

def config():
    st.markdown("""
        <style>
            .block-container {
                max-width: 1200px;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            p {
                text-align: justify;
            }
        </style>
    """, unsafe_allow_html=True)



# --- HELPER FUNCTIONS --- #
def load_model_and_scaler(model_name):
    model_path = model_options[model_name]
    model = joblib.load(model_path)
    
    scaler_path = SCALER_PATHS.get(model_name)
    scaler = joblib.load(scaler_path) if scaler_path else None
    return model, scaler


def load_manifest():
    try:
        with open("../nasa-apps-backend/metadata/manifest.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("manifest.json not found in models directory.")
        return {"models": []}


def display_model_metrics_sidebar(model_name, manifest):
    model_info = next(
        (m for m in manifest["models"] if m["name"].lower() == model_name.lower().replace(" ", "")),
        None
    )
    if not model_info:
        st.sidebar.warning("No metric data found for this model in manifest.json.")
        return

    if model_info.get("status") == "pending":
        st.sidebar.warning("This model is still being trained and evaluated.")
        return

    # Sidebar expander
    with st.sidebar.expander("Show Model Metrics", expanded=False):
        # Metrics table
        metrics_html = f"""
        <table style="width:100%; font-size:12px; text-align:center;">
            <tr>
                <th>Acc</th>
                <th>Prec</th>
                <th>Rec</th>
                <th>F1</th>
                <th>ROC</th>
            </tr>
            <tr>
                <td>{model_info['accuracy']:.3f}</td>
                <td>{model_info['precision']:.3f}</td>
                <td>{model_info['recall']:.3f}</td>
                <td>{model_info['f1_score']:.3f}</td>
                <td>{model_info['roc_auc']:.3f}</td>
            </tr>
        </table>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)

        # Heatmap
        cm = np.array(model_info["confusion_matrix"]["matrix"])
        display_small_heatmap(cm, container=st)  # pass container

def display_small_heatmap(cm, container=st):
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["F", "T"],
        yticklabels=["F", "T"],
        annot_kws={"size": 8, "color": "white"},
        linewidths=0.5,
        linecolor='black',
        ax=ax
    )

    ax.tick_params(axis='x', labelsize=8, colors='white')
    ax.tick_params(axis='y', labelsize=8, colors='white')
    ax.set_xlabel("Predicted", color="white", fontsize=9)
    ax.set_ylabel("Actual", color="white", fontsize=9)

    # Full black border
    rect = patches.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        color='black',
        linewidth=1.2,
        clip_on=False
    )
    ax.add_patch(rect)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight', pad_inches=0.1, transparent=True)
    buf.seek(0)
    container.image(buf)  # render in the container passed
    plt.close(fig)

def run_predictions(model, scaler, uploaded_file):
    candidates_df = pd.read_csv(uploaded_file)

    candidates_scaled = scaler.transform(candidates_df) if scaler else candidates_df
    preds = model.predict(candidates_scaled)

    results = candidates_df.copy()
    results["Prediction"] = preds.astype(int)  # Prediction stays int

    cols = ["Prediction"] + [c for c in results.columns if c != "Prediction"]
    results = results[cols]
    st.session_state["results"] = results

    # Display predictions directly without styling
    st.markdown("### Prediction Results")
    st.dataframe(results)

    # Download button
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")

    # --- Filter only confirmed exoplanets ---
    confirmed_df = results[results["Prediction"] == 1].copy()
    if confirmed_df.empty:
        st.warning("No confirmed exoplanets found in this dataset.")
    else:
        # --- Save CSV of confirmed only ---
        st.markdown("### Confirmed Exoplanets")
        st.dataframe(confirmed_df)
        csv_confirmed = confirmed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Confirmed Exoplanets CSV",
            data=csv_confirmed,
            file_name="confirmed_exoplanets.csv",
            mime="text/csv"
        )

    return results, confirmed_df


def display_visualizations(results, confirmed_df):

    if results.empty or "Prediction" not in results.columns:
        st.warning("No prediction data found.")
        return

    label_map = {1: "Confirmed Exoplanet", 0: "False Positive"}
    results["Label"] = results["Prediction"].map(label_map)
    counts = results["Label"].value_counts()

    st.markdown("### Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Candidates", len(results))
    col2.metric("Confirmed Exoplanets", counts.get("Confirmed Exoplanet", 0))
    col3.metric("False Positives", counts.get("False Positive", 0))

    col1, col2 = st.columns(2)
    with col1:
        # Bar chart
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(counts.index, counts.values, color=sns.color_palette("viridis", len(counts)))
        ax.set_ylabel("Count")
        ax.set_xticklabels(counts.index, rotation=20)
        st.pyplot(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
        ax2.axis("equal")
        st.pyplot(fig2, use_container_width=True)

    # Confirmed exoplanets by type
    if not confirmed_df.empty:
        confirmed_df["Exoplanet_Type"] = confirmed_df.apply(classify_exoplanet_type, axis=1)
        type_counts = confirmed_df["Exoplanet_Type"].value_counts()

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.barh(type_counts.index, type_counts.values,
                 color=sns.color_palette("viridis", len(type_counts)),
                 height=0.6, edgecolor='black')
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



def main():
    config()
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" 
                style="width:120px; height:120px; margin-right:10px;">
        </div>
        """,
        unsafe_allow_html=True
    )
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))

    manifest = load_manifest()
    try:
        model, scaler = load_model_and_scaler(selected_model)
        st.sidebar.success(f"{selected_model} loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading {selected_model}: {e}")
        return

    # Show metrics + heatmap above upload button
    display_model_metrics_sidebar(selected_model, manifest)

    # Sidebar CSV upload and Run button
    uploaded_file = st.sidebar.file_uploader("Upload CSV for prediction", type=["csv"])
    if uploaded_file is not None and st.sidebar.button("Run Model"):
        (results, confirmed_df) = run_predictions(model, scaler, uploaded_file)
        with st.expander("View Prediction Visualizations and Metrics", expanded=True):
            display_visualizations(results, confirmed_df)
    else:
        # If no results yet, show a friendly message
        st.markdown("""
            <style>
                .full-height {
                    display: flex;
                    justify-content: center;   /* horizontal center */
                    align-items: center;       /* vertical center */
                    height: 80vh;              /* 80% of viewport height */
                    color: gray;
                    font-size: 18px;
                }
            </style>
            <div class="full-height">
                Please upload a CSV file and click 'Run Model' to see predictions and visualizations.
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
