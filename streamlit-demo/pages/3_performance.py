import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("Model Performance")

# --------------------------
# Accuracy Section
# --------------------------
st.header("Accuracy Summary")
st.markdown("""
- **Primary Classifier (Fake vs Real)**: **97% Accuracy**
- **Secondary Classifier (Fake Subtypes)**: **86% Accuracy**
""")

# --------------------------
# Primary Classifier
# --------------------------
st.header("Primary Classifier Confusion Matrix")
st.markdown("This classifier distinguishes between **Fake** and **Real** news.")

primary_data = {
    "Real (0)": [0.88, 0.00],
    "Fake (1)": [0.12, 1.00],
}
primary_df = pd.DataFrame(primary_data, index=["Real (0)", "Fake (1)"])
st.dataframe(primary_df.style.format("{:.2f}").highlight_max(axis=1, color='darkgreen'), use_container_width=True)

# --------------------------
# Secondary Classifier
# --------------------------
st.header("Secondary Classifier Confusion Matrix")
st.markdown("This classifier identifies the **type of fake news**, classifying it into one of the following categories:")

labels = ["bias", "clickbait", "conspiracy", "fake", "hate", "junksci", "political", "rumor", "satire", "unreliable"]
secondary_data = [
    [0.78, 0.00, 0.02, 0.00, 0.00, 0.15, 0.02, 0.02, 0.02, 0.00],
    [0.02, 0.87, 0.00, 0.00, 0.02, 0.03, 0.00, 0.02, 0.07, 0.00],
    [0.00, 0.00, 0.85, 0.00, 0.05, 0.05, 0.03, 0.00, 0.03, 0.00],
    [0.00, 0.02, 0.05, 0.86, 0.00, 0.02, 0.02, 0.00, 0.04, 0.00],
    [0.02, 0.02, 0.00, 0.00, 0.94, 0.00, 0.00, 0.00, 0.00, 0.02],
    [0.00, 0.00, 0.05, 0.00, 0.02, 0.93, 0.00, 0.00, 0.00, 0.00],
    [0.02, 0.07, 0.03, 0.02, 0.00, 0.03, 0.78, 0.00, 0.03, 0.02],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.93, 0.04, 0.00],
    [0.00, 0.02, 0.04, 0.00, 0.00, 0.13, 0.00, 0.02, 0.78, 0.00],
    [0.06, 0.04, 0.00, 0.00, 0.00, 0.02, 0.00, 0.04, 0.02, 0.83],
]

secondary_df = pd.DataFrame(secondary_data, index=labels, columns=labels)
st.dataframe(secondary_df.style.format("{:.2f}").highlight_max(axis=1, color='purple'), use_container_width=True)

st.markdown("---")
st.caption("Performance metrics based on evaluation on held-out test datasets.")
