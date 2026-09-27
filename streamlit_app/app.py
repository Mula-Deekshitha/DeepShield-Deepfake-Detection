import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
import sys

# Ensure project root is in system path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit_app.local_inference import LocalInferencePipeline

st.set_page_config(
    page_title="DEEPSHIELD - Deepfake Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ DEEPSHIELD: REAL-TIME DEEPFAKE DETECTION ENGINE")
st.markdown("### CNN-Vision Transformer Fusion with Temporal & Multi-Modal Analysis")

# Sidebar configurations
st.sidebar.title("⚙️ Engine Configurations")
st.sidebar.info("Runtime Context: Local Windows CPU (INT8 Quantized ONNX)")
threshold = st.sidebar.slider("Deepfake Sensitivity Threshold", 0.0, 1.0, 0.50)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Specifications:**")
st.sidebar.text("• Multi-Modal Feature Fusion\n• 468 Landmark Mesh GCN\n• FFT Frequency Analysis\n• Grad-CAM++ Explanations")

# File Upload Section
uploaded_file = st.file_uploader("Upload Target Video File (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to temporary directory
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📹 Input Video Stream")
        st.video(tfile.name)
        
    with col2:
        st.subheader("📊 Diagnostic & Anomaly Engine")
        run_btn = st.button("🚀 Analyze Video Stream", use_container_width=True)
        
        if run_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            status_text.text("Extracting spatial video frames & 468 3D landmarks...")
            progress_bar.progress(30)
            
            status_text.text("Computing Fast Fourier Transforms (FFT) & Frequency Maps...")
            progress_bar.progress(60)
            
            # Execute ONNX Local CPU Inference
            pipeline = LocalInferencePipeline()
            status_text.text("Executing ONNX Quantized Fusion Engine on CPU...")
            
            fake_probability, heatmap = pipeline.predict(tfile.name)
            progress_bar.progress(100)
            
            latency = time.time() - start_time
            status_text.text("Analysis Complete!")
            
            st.divider()
            
            # Prediction Results
            if fake_probability >= threshold:
                st.error("⚠️ **FINAL VERDICT: DEEPFAKE / MANIPULATED VIDEO DETECTED**")
            else:
                st.success("✅ **FINAL VERDICT: AUTHENTIC / REAL VIDEO DETECTED**")
                
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="Deepfake Confidence Score", value=f"{fake_probability * 100:.2f}%")
            with m_col2:
                st.metric(label="Total CPU Inference Latency", value=f"{latency:.2f} sec")
                
            st.divider()
            st.subheader("🔥 Grad-CAM++ Manipulated Region Heatmap")
            
            # Render Heatmap Overlay
            heatmap_uint8 = np.uint8(255 * heatmap)
            colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
            
            st.image(
                colored_heatmap, 
                caption="Highlighted Tampering Anomalies (Eyes, Lips, and Facial Contours)", 
                width=320
            )

    # Clean up temp file safely after processing
    try:
        os.remove(tfile.name)
    except Exception:
        pass