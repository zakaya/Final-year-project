import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import google.generativeai as genai

# ---- Directly assign your Gemini API key here ----
GEMINI_API_KEY = "AIzaSyCssg4lzrC9ziamYLL3BXe2H5dS7ztWhZc"  # <-- Your actual API key here

if not GEMINI_API_KEY:
    st.error("Gemini API Key not found. Please set the GEMINI_API_KEY variable in your code.")
    st.stop()

# ---- Custom CSS ----
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8ffae 0%, #43c6ac 100%);
    }
    .main > div {
        background-color: rgba(255,255,255,0.85);
        border-radius: 20px;
        padding: 2em;
        margin-top: 2em;
    }
    </style>
    """, unsafe_allow_html=True)

# ---- Sidebar content ----
st.sidebar.image("https://img.icons8.com/color/96/000000/bone.png", width=100)
st.sidebar.title("About")
st.sidebar.info("""
Bone Fracture Classifier  
Upload an X-ray image to detect bone fractures using an AI model.

- *Author:* Your Name
- *Contact:* your.email@example.com
""")
st.sidebar.markdown("---")
st.sidebar.write("Made with ❤ using Streamlit")

# ---- Main Title ----
st.markdown('<h1 class="title">🦴 Bone Fracture Classifier</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an X-ray image to check for fractures</div>', unsafe_allow_html=True)

# ---- Load model ----
@st.cache_resource
def load_my_model():
    return load_model('my_model.h5')

model = load_my_model()

def preprocess_image(image: Image.Image):
    img = image.resize((299, 299))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

prediction = None
label = ""
confidence = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray.", width=300)
    st.write("Classifying...")

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]
    label = " ✅Not Fractured" if prediction > 0.5 else " 🩹Fractured"
    color = "green" if prediction > 0.5 else "red"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"<h2 style='color:{color};'>Prediction: {label}</h2>", unsafe_allow_html=True)
    st.markdown(f"*Model Confidence:* {prediction:.2%}")

    st.progress(float(confidence))

    # ---- Gemini Integration Section ----
    genai.configure(api_key=GEMINI_API_KEY)
    st.markdown("---")
    st.header("💬 Ask the chatbot about Bone Fractures")

    default_prompt = (
        f"The uploaded X-ray was classified as NOT FRACTURED with {prediction:.2%} confidence. "
        "Explain what this means for the patient, and what further steps or care may be needed."
        if prediction > 0.5 else
        f"The uploaded X-ray was classified as FRACTURED with {(1-prediction):.2%} confidence. "
        "Explain what this means for the patient, possible bone fracture types, and what actions are recommended."
    )

    user_prompt = st.text_area(
        "Ask a medical or X-ray-related question (or let Gemini explain the prediction):",
        value=default_prompt,
        height=100,
    )

    if st.button("Ask Gemini") and user_prompt.strip():
        with st.spinner("Gemini is thinking..."):
            try:
                gmodel = genai.GenerativeModel("models/gemini-1.5-flash-latest")
                response = gmodel.generate_content(user_prompt)
                st.markdown("#### Gemini says:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Gemini API Error: {e}")

else:
    st.info("Please upload an X-ray image to begin.")

# ---- Footer ----
st.markdown("""
    <hr>
    <center>
    <small>Developed with Streamlit | <a href="mailto:your.email@example.com">Contact</a></small>
    </center>
    """, unsafe_allow_html=True)