# Predicción de concentración de monóxido de carbono con Machine Learning

## Descripción del proyecto
Este proyecto utiliza técnicas de Machine Learning para predecir la concentración de monóxido de carbono CO(GT) en el aire, usando el dataset Air Quality de UCI Machine Learning Repository.  
El objetivo es aplicar un modelo de regresión capaz de estimar valores numéricos a partir de variables ambientales y sensores químicos.

## Problema que resuelve
La contaminación del aire es un problema importante porque puede afectar la salud de las personas.  
Este sistema busca apoyar el análisis de calidad del aire mediante la predicción de CO usando Inteligencia Artificial.

## Tipo de problema
Este proyecto es de regresión, porque el modelo no clasifica en categorías, sino que predice un valor numérico continuo.

## Dataset
- Nombre: Air Quality
- Fuente: UCI Machine Learning Repository
- Registros: 9,358
- Variables: 15
- Variable objetivo: CO(GT)

## Tecnologías utilizadas
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- TensorFlow / Keras
- Streamlit

## Modelos utilizados
- Regresión Lineal
- Random Forest Regressor
- Red Neuronal Dense

## Métricas de evaluación
El modelo se evalúa usando métricas de regresión:

- MAE
- MSE
- RMSE
- R²

## Estructura del proyecto

proyecto-ia-calidad-aire/
│
├── app.py
├── AirQualityUCI.csv
├── modelo_co.keras
├── imputador.pkl
├── escalador.pkl
├── columnas.pkl
├── requirements.txt
├── Proyecto_Final_Air_Quality_ML.ipynb
└── README.md

## Instalación

Para instalar las librerías necesarias, ejecutar:

```bash
pip install -r requirements.txt