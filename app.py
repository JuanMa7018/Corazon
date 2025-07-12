#App para predicción de problema cardiaco
#NO CORRERO CON EL BOTON DE RUN ***********************

!pip install streamlit joblib scikit-learn pandas matplotlib seaborn

import streamlit as st
import joblib
import pandas as pd
import os

# Load the trained model and scaler
try:
    svc_model = joblib.load('best_svc_model.jb')
    scaler = joblib.load('minmax_scaler.jb')
except FileNotFoundError:
    st.error("Error loading model or scaler. Make sure 'best_svc_model.jb' and 'minmax_scaler.jb' are in the same directory.")
    st.stop()


# Function to make predictions
def predict_heart_problem(age, cholesterol):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({'edad': [age], 'colesterol': [cholesterol]})

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = svc_model.predict(scaled_input)

    return prediction[0]

# --- Streamlit App ---

# Load images
try:
    cabezote_image = 'cabezote.jpeg'
    nosufre_image = 'nosufre.jpeg'
    sisufre_image = 'sisufre.jpeg'
except FileNotFoundError:
    st.error("Image files not found. Please ensure 'cabezote.jpeg', 'nosufre.jpeg', and 'sisufre.jpeg' are in the same directory.")
    cabezote_image = None
    nosufre_image = None
    sisufre_image = None


# Add banner image
if cabezote_image and os.path.exists(cabezote_image):
    st.image(cabezote_image, use_column_width=True)
else:
    st.warning("Banner image not found.")


st.title("Modelo IA para predicción de problemas cardiacos")

st.markdown("""
Este modelo de Inteligencia Artificial utiliza un clasificador de Máquinas de Vectores de Soporte (SVC)
entrenado para predecir la probabilidad de sufrir problemas cardíacos basado en la edad y los niveles de colesterol.
El modelo ha sido entrenado y validado para ofrecer una predicción informada.
""")

# Sidebar for user input
st.sidebar.header("Ingresa los datos del paciente")

age = st.sidebar.slider("Edad:", min_value=20, max_value=80, value=20, step=1)
cholesterol = st.sidebar.slider("Colesterol:", min_value=120, max_value=600, value=200, step=10)

st.write("---")

# Make prediction when button is clicked
if st.button("Predecir"):
    prediction = predict_heart_problem(age, cholesterol)

    st.header("Resultado de la Predicción")

    if prediction == 0:
        st.markdown("<div style='background-color:#90EE90; padding: 10px; border-radius: 5px; color: black;'>", unsafe_allow_html=True)
        st.markdown("<h3>0: ¡No sufrirá del corazón! 😊</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if nosufre_image and os.path.exists(nosufre_image):
            st.image(nosufre_image, use_column_width=True)
        else:
            st.warning("Image 'nosufre.jpg' not found.")

    else:
        st.markdown("<div style='background-color:#F08080; padding: 10px; border-radius: 5px; color: black;'>", unsafe_allow_html=True)
        st.markdown("<h3>1: Posiblemente sufrirá del corazón 😞</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if sisufre_image and os.path.exists(sisufre_image):
            st.image(sisufre_image, use_column_width=True)
        else:
            st.warning("Image 'Sisure.jpg' not found.")

st.write("---")

st.markdown("Elaborado por: Alfredo Diaz © Unab 2025")
