import pyspark.sql.functions as F
import logging


class DataExplorer:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.logger = logging.getLogger(self.__class__.__name__)

    def show_info(self):
        """Muestra la información general del DataFrame."""
        try:
            self.logger.info("Obteniendo información general del DataFrame")
            self.dataframe.printSchema()
            self.logger.info("Información general obtenida con éxito")
        except Exception as e:
            self.logger.error(f"Error al obtener la información del DataFrame: {e}")

    def show_describe(self):
        """Muestra estadísticas descriptivas del DataFrame."""
        try:
            self.logger.info("Obteniendo estadísticas descriptivas del DataFrame")
            self.dataframe.describe().show()
            self.logger.info("Estadísticas descriptivas obtenidas con éxito")
        except Exception as e:
            self.logger.error(f"Error al obtener estadísticas descriptivas: {e}")

    def show_null_values(self):
        """Muestra el conteo de valores nulos en el DataFrame."""
        try:
            self.logger.info("Calculando valores nulos en el DataFrame")
            null_values = self.dataframe.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in self.dataframe.columns])
            null_values.show()
            self.logger.info("Valores nulos calculados con éxito")
        except Exception as e:
            self.logger.error(f"Error al calcular valores nulos: {e}")