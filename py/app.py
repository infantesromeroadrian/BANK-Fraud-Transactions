# streamlit_app.py

import streamlit as st
import pandas as pd
import mlflow.spark
from pyspark.sql import SparkSession

# Inicializar Spark Session
spark = SparkSession.builder.appName("FraudDetectionApp").getOrCreate()

# Función para cargar el modelo
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = mlflow.spark.load_model(model_path)
    return model

# Función para hacer predicciones
def predict(model, data):
    df = spark.createDataFrame(data)
    predictions = model.transform(df)
    return predictions.select('prediction').collect()[0][0]

# Cargar el modelo (ajustar el nombre del modelo según sea necesario)
model_path = "/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/Portfolio/BankProjects/FraudTransactionsDetection/model/model_RandomForest"
model = load_model(model_path)

# Interfaz de Streamlit
st.title("Detector de Fraude en Transacciones Bancarias")

# Crear entradas para datos de transacción
type = st.selectbox("Tipo de Transacción", ['TRANSFER', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'CASH_IN'])
amount = st.number_input("Monto de la Transacción", min_value=0.0, step=1.0)
oldbalanceOrg = st.number_input("Saldo Inicial del Emisor", min_value=0.0, step=1.0)
newbalanceOrig = st.number_input("Saldo Final del Emisor", min_value=0.0, step=1.0)

# Añadir otras entradas necesarias para tu modelo aquí...

# Botón para realizar predicción
if st.button('Predecir'):
    # Crear DataFrame con los valores de entrada
    data = pd.DataFrame({
        'type': [type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        # Añadir otras columnas necesarias para tu modelo...
    })

    # Realizar predicción
    prediction = predict(model, data)
    if prediction == 1:
        st.error("La transacción es sospechosa de ser un FRAUDE.")
    else:
        st.success("La transacción parece LEGÍTIMA.")

# Ejecutar con: streamlit run streamlit_app.py
