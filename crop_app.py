import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# -------- SAFE FILE LOADING --------
BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "crop_model.pkl")

# Try loading encoder only if it exists
label_encoder = None
encoder_path = BASE_DIR / "label_encoder.pkl"
if encoder_path.exists():
    label_encoder = joblib.load(encoder_path)

# -------- UI --------
st.set_page_config(page_title="Smart Crop Recommendation", layout="centered")

st.title("🌾 Smart Crop Recommendation System")
st.write("### Enter soil & weather details to get best crop suggestion")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen", 0, 200, 50)
    P = st.number_input("Phosphorus", 0, 200, 50)
    K = st.number_input("Potassium", 0, 200, 50)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

with col2:
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

if st.button("🌿 Recommend Crop"):
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    pred = model.predict(data)

    # If encoder exists → decode
    if label_encoder:
        crop = label_encoder.inverse_transform(pred)[0]
    else:
        crop = pred[0]

    st.success(f"### ✅ Recommended Crop: **{crop.upper()}** 🌱")
    st.balloons()

    st.info("""
### Benefits
✔ Helps farmers  
✔ Increases productivity  
✔ Smart Decision Support System  
""")

st.markdown("---")
st.caption("Developed with ❤️ using Machine Learning & Streamlit")
