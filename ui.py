import streamlit as st
import matplotlib.pyplot as plt

def show_header():
    """Display app title and description."""
    st.title("🧠 AI-Generated Image Detector")
    st.markdown("""
    **Upload an image or enter an image URL** to detect whether it’s AI-generated or real.  
    The model is based on **EfficientNet-B3**, fine-tuned for image authenticity detection.
    """)


def show_image(img, label, prob):
    """Display image and prediction result."""
    st.image(img, caption="Uploaded Image", use_container_width=True)

    st.subheader("Prediction Results")
    st.write(f"**Label:** {label}")
    
    # Optional: display confidence bar
    st.progress(min(int(prob * 100), 100))
