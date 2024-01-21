# data_ingestor.py

import logging
from pyspark.sql import SparkSession

# Configuración del Logger
logging.basicConfig(level=logging.INFO)

class DataIngestor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.spark = self.start_spark_session()

    def start_spark_session(self):
        """Iniciar una sesión Spark."""
        self.logger.info("Iniciando sesión Spark")
        return SparkSession.builder.appName("IoT Intrusion Detection").getOrCreate()

    def read_data(self):
        """Leer datos desde un archivo CSV."""
        self.logger.info(f"Leyendo datos desde {self.file_path}")
        df = self.spark.read.csv(self.file_path, header=True, inferSchema=True)
        self.logger.info("Lectura del archivo completada")
        return df
