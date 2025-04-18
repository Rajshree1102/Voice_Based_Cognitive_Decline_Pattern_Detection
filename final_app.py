import streamlit as st
import os
import numpy as np
from cognitive_decline_detection import detect_cognitive_decline

st.set_page_config(page_title="Cognitive Decline Detection", layout="centered")

st.title("Voice-Based Cognitive Decline Detection")
st.markdown("Upload a `.wav` file to analyze cognitive speech patterns for early decline detection.")

uploaded_file = st.file_uploader("🎙 Upload an audio file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file
    os.makedirs("temp_audio", exist_ok=True)
    file_path = os.path.join("temp_audio", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Analyzing audio..."):
        try:
            result = detect_cognitive_decline(file_path)

            # Extract values
            text = result["audio_to_text"]
            score = float(result["risk_score"])
            features = result["features"]

            # Convert numpy types to native types
            features = {
                k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
                for k, v in features.items()
            }

            # Show results
            st.success("✅ Analysis Complete!")
            st.subheader("📝 Transcribed Text")
            st.write(text)

            st.subheader("📊 Cognitive Risk Score")
            st.metric(label="Risk Score", value=round(score, 3))

            st.subheader("🔬 Extracted Features")
            st.json(features)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
