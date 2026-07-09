import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# ===============================
# CARGAR MODELO Y PREPROCESADORES
# ===============================

with open("imputador.pkl", "rb") as f:
    imputador = pickle.load(f)

with open("escalador.pkl", "rb") as f:
    escalador = pickle.load(f)

with open("columnas.pkl", "rb") as f:
    columnas = pickle.load(f)

# ===============================
# INTERFAZ WEB
# ===============================

st.set_page_config(
    page_title="Predicción de CO",
    page_icon="🌫️",
    layout="centered"
)

st.title("🌫️ Predicción de Monóxido de Carbono")
st.write("""
Esta página utiliza un modelo de Inteligencia Artificial para predecir la concentración de **CO(GT)** 
a partir de variables de sensores químicos y condiciones ambientales.
""")

st.subheader("Ingresa los valores de entrada")

datos_usuario = {}

for columna in columnas:
    datos_usuario[columna] = st.number_input(
        label=columna,
        value=0.0,
        step=0.1
    )

# ===============================
# BOTÓN DE PREDICCIÓN
# ===============================

if st.button("Predecir concentración de CO"):
    
    # Convertir datos del usuario en DataFrame
    entrada = pd.DataFrame([datos_usuario])

    # Aplicar la misma preparación usada en el entrenamiento
    entrada_imputada = imputador.transform(entrada)
    entrada_escalada = escalador.transform(entrada_imputada)

    # Realizar predicción
    prediccion = modelo.predict(entrada_escalada)[0][0]

    st.success(f"Predicción estimada de CO(GT): {prediccion:.4f}")

    st.write("""
    **Interpretación:**  
    Este valor representa la concentración estimada de monóxido de carbono según los datos ingresados.
    Mientras más bajo sea el error del modelo durante la evaluación, más confiable será esta predicción.
    """)
