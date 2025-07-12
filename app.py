#App para predicción de problema cardiaco
#NO CORRERO CON EL BOTON DE RUN ***********************

!pip install streamlit joblib scikit-learn pandas

import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo y el scaler
# Asegúrate de que los archivos 'bert_svc_model.jb' y 'minmax_scaler.jb'
# estén en el mismo directorio que este script o proporciona la ruta completa.
# Si guardaste con .jb, cámbialo aquí
try:
    svc_model = joblib.load('best_svc_model.jb')
    scaler = joblib.load('minmax_scaler.jb') # Asegúrate de haber guardado el scaler también
except FileNotFoundError:
    st.error("Archivos de modelo o scaler no encontrados. Asegúrate de que 'svc_model.pkl' y 'scaler.pkl' están presentes.")
    st.stop()

# Título de la aplicación
st.image('cabezote.jpeg', use_column_width=True)
st.title("Modelo IA para predicción de problemas cardiacos")

# Resumen del modelo
st.markdown("""
Este modelo utiliza Inteligencia Artificial para predecir la probabilidad de que un paciente sufra de problemas cardiacos,
basándose en su edad y niveles de colesterol. El modelo fue entrenado utilizando el algoritmo de Máquinas de Vectores de Soporte (SVC)
y los datos de entrada fueron escalados para mejorar la precisión.
""")

# Sidebar con sliders
st.sidebar.header("Ingrese los datos del paciente")

edad = st.sidebar.slider("Edad", 20, 80, 20, 1)
colesterol = st.sidebar.slider("Colesterol", 120, 600, 200, 10)

# Preparar los datos para la predicción
input_data = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})

# Escalar los datos de entrada
input_data_scaled = scaler.transform(input_data)

# Realizar la predicción
prediction = svc_model.predict(input_data_scaled)

# Mostrar el resultado
st.header("Resultado de la Predicción")

if prediction[0] == 0:
    st.markdown(
        "<div style='background-color:#90ee90; padding: 10px; border-radius: 5px;'>"
        "<h3>0: No sufrirá del corazón 😊</h3>"
        "</div>", unsafe_allow_html=True
    )
    st.image('nosufre.jpg', use_column_width=True)
else:
    st.markdown(
        "<div style='background-color:#ff6347; padding: 10px; border-radius: 5px;'>"
        "<h3>1: Sufrirá del corazón 😟</h3>"
        "</div>", unsafe_allow_html=True
    )
    st.image('sisufre.jpg', use_column_width=True)

# Información del elaborador
st.markdown("---")
st.markdown("Elaborado por: Alfredo Diaz © Unab 2025")
