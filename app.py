import streamlit as st
from utils import load_model, load_image, preprocess_image, predict
from ui import show_header, show_image
import os

# ========================================
# 🔧 Configuration
# ========================================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_b3_full_ai_image_classifier.pt")

# ========================================
# 🚀 Streamlit App
# ========================================
def main():
    st.set_page_config(page_title="AI Image Detector", page_icon="🧠", layout="centered")
    show_header()

    # Load model once and cache
    @st.cache_resource
    def get_model():
        return load_model(MODEL_PATH)

    model = get_model()

    # User options
    option = st.radio("Choose Input Type:", ("Upload Image", "From URL"))

    img = None
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = load_image(uploaded_file)
    else:
        url = st.text_input("Enter Image URL")
        if url:
            img = load_image(url)

    # Predict
    if img is not None:
        img_tensor = preprocess_image(img)
        label, prob = predict(model, img_tensor)
        show_image(img, label, prob)
    else:
        st.info("👆 Upload an image or enter a URL to start.")


if __name__ == "__main__":
    main()
