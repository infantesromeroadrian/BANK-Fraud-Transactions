from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col
from pyspark.sql.types import StringType
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
import logging

class Splitter:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.logger = logging.getLogger(self.__class__.__name__)

    def select_features(self, feature_columns, label_column):
        """Selecciona las columnas de características y etiquetas para el modelo."""
        try:
            self.logger.info("Seleccionando características y etiqueta")
            # Asegurar que la columna de etiqueta es de tipo String para la indexación
            if isinstance(label_column, str):
                self.dataframe = self.dataframe.withColumn(label_column, col(label_column).cast(StringType()))

            # Usar VectorAssembler para combinar las columnas de características
            assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
            self.dataframe = assembler.transform(self.dataframe)

            # Indexar la columna de etiqueta
            label_indexer = StringIndexer(inputCol=label_column, outputCol="label")
            self.dataframe = label_indexer.fit(self.dataframe).transform(self.dataframe)

            self.logger.info("Características y etiqueta seleccionadas con éxito")
            return self.dataframe
        except Exception as e:
            self.logger.error(f"Error al seleccionar características y etiqueta: {e}")
            raise

    def split_data(self, train_ratio=0.7):
        """Divide los datos en conjuntos de entrenamiento y prueba."""
        try:
            self.logger.info("Dividiendo los datos en conjuntos de entrenamiento y prueba")
            train_data, test_data = self.dataframe.randomSplit([train_ratio, 1 - train_ratio], seed=42)
            self.logger.info("División de datos completada con éxito")
            return train_data, test_data
        except Exception as e:
            self.logger.error(f"Error al dividir los datos: {e}")
            raise