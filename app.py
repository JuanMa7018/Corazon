#App para predicción de problema cardiaco
#NO CORRERO CON EL BOTON DE RUN ***********************

!pip install streamlit joblib scikit-learn pandas matplotlib seaborn

import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Cargar el modelo y el escalador
try:
    svc_model = joblib.load('best_svc_model.jb')
    scaler = joblib.load('minmax_scaler.jb')
except FileNotFoundError:
    st.error("Error: Los archivos del modelo (best_svc_model.jb) o del escalador (minmax_scaler.jb) no fueron encontrados.")
    st.stop()

# Título de la aplicación
st.title("Modelo IA para predicción de problemas cardiacos")

# Imagen de banner (asegúrate de que cabezote.jpeg esté en la misma carpeta o especifica la ruta completa)
try:
    st.image("cabezote.jpeg", use_column_width=True)
except FileNotFoundError:
    st.warning("Advertencia: La imagen 'cabezote.jpeg' no fue encontrada. Asegúrate de que esté en la misma carpeta.")


# Resumen del modelo para el usuario
st.markdown("""
Este modelo de Inteligencia Artificial utiliza técnicas de Machine Learning para predecir la probabilidad de que un paciente pueda sufrir problemas cardiacos basándose en su edad y nivel de colesterol.

El modelo fue entrenado utilizando datos históricos de pacientes y un algoritmo llamado **Support Vector Machine (SVM)**. Antes de entrenar, los datos de edad y colesterol fueron ajustados usando un método de escalado para mejorar el rendimiento del modelo.

Introduce la edad y el nivel de colesterol en el panel izquierdo para obtener una predicción.
""")

# Sidebar para la entrada de datos del usuario
st.sidebar.header("Introduce los datos del paciente")

edad = st.sidebar.slider("Edad:", min_value=20, max_value=80, value=20, step=1)
colesterol = st.sidebar.slider("Colesterol:", min_value=120, max_value=600, value=200, step=10)

# Preparar los datos de entrada para el modelo
data_input = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})

# Escalar los datos de entrada usando el scaler cargado
try:
    data_scaled = scaler.transform(data_input)
    data_scaled_df = pd.DataFrame(data_scaled, columns=['edad', 'colesterol'])

    # Realizar la predicción
    prediction = svc_model.predict(data_scaled_df)

    # Mostrar los resultados
    st.subheader("Resultado de la predicción:")

    if prediction[0] == 0:
        st.markdown(
            f"""
            <div style='background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px;'>
                <p>✅ **0: No sufrirá del corazón**</p>
            </div>
            """, unsafe_allow_html=True
        )
        try:
            st.image("nosufre.jpg", caption="¡Buenas noticias!", use_column_width=True)
        except FileNotFoundError:
            st.warning("Advertencia: La imagen 'nosufre.jpg' no fue encontrada.")
    else:
        st.markdown(
            f"""
            <div style='background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;'>
                <p>😟 **1: Sufrirá del corazón**</p>
            </div>
            """, unsafe_allow_html=True
        )
        try:
            st.image("Sisure.jpg", caption="Considera consultar a un médico.", use_column_width=True)
        except FileNotFoundError:
            st.warning("Advertencia: La imagen 'Sisure.jpg' no fue encontrada.")

except Exception as e:
    st.error(f"Ocurrió un error durante el procesamiento de los datos o la predicción: {e}")


# Información del elaborador
st.markdown("---")
st.markdown("Elaborado por: Alfredo Diaz © Unab 2025")
