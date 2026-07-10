import streamlit as st
import pandas as pd
import pickle

# ===============================
# CARGAR MODELO Y ARCHIVOS
# ===============================

with open("modelo_rf.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("imputador.pkl", "rb") as f:
    imputador = pickle.load(f)

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
Esta aplicación utiliza un modelo de Machine Learning para predecir la concentración de **CO(GT)** 
a partir de variables ambientales y sensores químicos.
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

    entrada = pd.DataFrame([datos_usuario])

    # Mantener el mismo orden de columnas usado en entrenamiento
    entrada = entrada[columnas]

    # Aplicar el mismo imputador usado en el notebook
    entrada_imputada = imputador.transform(entrada)

    # Predicción con Random Forest
    prediccion = modelo.predict(entrada_imputada)[0]

    st.success(f"Predicción estimada de CO(GT): {prediccion:.4f}")

    st.write("""
    **Interpretación:**  
    Este valor representa la concentración estimada de monóxido de carbono según los datos ingresados.
    """)
