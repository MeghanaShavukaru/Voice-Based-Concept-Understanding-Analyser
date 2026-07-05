import streamlit as st
import json
from pathlib import Path
from intelligence import *
from reporting import generate_pdf_report

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
REFERENCES_PATH = BASE_DIR / "references.json"

# Setup Temp Directory
if TEMP_DIR.exists() and not TEMP_DIR.is_dir():
    TEMP_DIR.unlink()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Load Reference Data
with open(REFERENCES_PATH, "r") as f:
    references = json.load(f)

st.set_page_config(page_title="VBCUA System", layout="wide")
st.title("🎙️ Voice-Based Concept Understanding Analyser")
st.markdown("Evaluate conceptual understanding and speech fluency via AI.")

# UI: Concept Selection
topic = st.selectbox("Select a Concept to Explain:", list(references.keys()))
st.info(f"**Target Concept:** {topic}")

# UI: Audio Upload
uploaded_file = st.file_uploader("Upload your audio explanation (WAV/MP3)", type=['wav', 'mp3'])

if uploaded_file is not None:
    # Save file temporarily
    audio_path = TEMP_DIR / uploaded_file.name
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.audio(str(audio_path), format="audio/wav")
    
    if st.button("Evaluate Performance"):
        with st.spinner("Analyzing audio and semantics... This may take a minute."):
            
            # 1. Speech to Text
            transcript = transcribe_audio(audio_path)
            
            # 2. Semantic Analysis
            semantic_score = analyze_semantics(transcript, references[topic])
            
            # 3. Audio & Fluency Features
            plot_path = TEMP_DIR / "waveform.png"
            rms_energy, pause_ratio = analyze_audio_features(audio_path, plot_path)
            filler_count, filler_ratio = analyze_filler_words(transcript)
            
            # 4. Scoring
            final_score, feedback = generate_score_and_feedback(semantic_score, pause_ratio, filler_ratio)
            if "Transcription unavailable" in transcript or "Unable to load" in transcript:
                feedback = "Analysis completed with limited AI support. Please check your network or model access."
            
            # Display Dashboard Metrics
            st.markdown("---")
            st.header("📊 Evaluation Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Score", f"{final_score:.1f}/100")
            col2.metric("Semantic Match", f"{semantic_score:.1f}%")
            col3.metric("Pause Ratio", f"{pause_ratio:.1f}%")
            col4.metric("Filler Words", f"{filler_ratio:.1f}%")
            
            st.success(f"**AI Assessment:** {feedback}")
            
            st.subheader("Transcription")
            st.write(f'"{transcript}"')
            
            st.subheader("Waveform Visualization")
            st.image(plot_path)
            
            # 5. Report Generation
            report_path = TEMP_DIR / "Evaluation_Report.pdf"
            generate_pdf_report(
                report_path, transcript, semantic_score, pause_ratio, 
                filler_ratio, rms_energy, final_score, feedback, plot_path
            )
            
            # PDF Download Button
            with open(report_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_file,
                    file_name="VBCUA_Report.pdf",
                    mime="application/pdf"
                )