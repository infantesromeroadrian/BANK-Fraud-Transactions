from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import to_date, col
import logging

class DataTransformer:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.logger = logging.getLogger(self.__class__.__name__)
        self.to_drop = []  # Inicialización del atributo to_drop

    def one_hot_encode(self, columns):
        """Aplica One-Hot Encoding a las columnas especificadas y elimina las columnas originales."""
        try:
            self.logger.info("Aplicando One-Hot Encoding")
            stages = []  # Etapas para el pipeline

            for col_name in columns:
                string_indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed")
                encoder = OneHotEncoder(inputCols=[string_indexer.getOutputCol()], outputCols=[col_name + "_encoded"])
                stages += [string_indexer, encoder]
                self.to_drop.append(col_name)

            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(self.dataframe)
            self.dataframe = model.transform(self.dataframe)

            self.dataframe = self.dataframe.drop(*self.to_drop)

            self.logger.info("One-Hot Encoding aplicado con éxito")
            return self.dataframe
        except Exception as e:
            self.logger.error(f"Error al aplicar One-Hot Encoding: {e}")
            raise

    def convert_to_date(self, column):
        """Convierte una columna a tipo fecha."""
        try:
            self.logger.info(f"Convirtiendo la columna {column} a tipo fecha")
            # Usando el formato 'dd-MMM-yy' para la conversión
            self.dataframe = self.dataframe.withColumn(column, to_date(col(column), 'dd-MMM-yy'))
            self.logger.info(f"Columna {column} convertida a tipo fecha con éxito")
            return self.dataframe
        except Exception as e:
            self.logger.error(f"Error al convertir la columna {column} a tipo fecha: {e}")
            raise

    def scale_features(self, input_cols):
        """Escala las características numéricas especificadas y elimina las columnas originales."""
        try:
            self.logger.info("Aplicando escalado a las características numéricas")
            assembler = VectorAssembler(inputCols=input_cols, outputCol="features_to_scale")
            scaler = StandardScaler(inputCol="features_to_scale", outputCol="scaled_features")

            self.dataframe = assembler.transform(self.dataframe)
            scaler_model = scaler.fit(self.dataframe)
            self.dataframe = scaler_model.transform(self.dataframe)

            self.dataframe = self.dataframe.drop("features_to_scale")

            self.logger.info("Escalado de características completado con éxito")
            return self.dataframe
        except Exception as e:
            self.logger.error(f"Error al aplicar el escalado de características: {e}")
            raise