import logging
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
import logging

class ModelTrainer:
    def __init__(self, train_data, test_data, models):
        """
        Inicializa la clase ModelTrainer.
        :param train_data: DataFrame de Spark para entrenamiento.
        :param test_data: DataFrame de Spark para pruebas.
        :param models: Diccionario de modelos a entrenar con sus nombres como claves.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.models = models
        self.logger = logging.getLogger(self.__class__.__name__)
        self.best_models = {}

    def fit_and_evaluate(self):
        """
        Entrena y evalúa cada modelo en el diccionario de modelos.
        """
        for name, model_info in self.models.items():
            try:
                self.logger.info(f"Entrenando y evaluando modelo: {name}")
                model = model_info['model']
                param_grid = model_info['param_grid']

                # Iniciar un experimento MLflow
                with mlflow.start_run(run_name=name) as run:
                    # Crear Pipeline
                    pipeline = Pipeline(stages=[model])

                    # Crear Grid de Parámetros
                    grid = ParamGridBuilder()
                    for param, values in param_grid.items():
                        grid = grid.addGrid(param, values)
                    param_grid = grid.build()

                    # Crear y ejecutar CrossValidator
                    evaluator = BinaryClassificationEvaluator()
                    crossval = CrossValidator(estimator=pipeline,
                                              estimatorParamMaps=param_grid,
                                              evaluator=evaluator,
                                              numFolds=3)

                    cv_model = crossval.fit(self.train_data)

                    # Evaluar en el conjunto de prueba
                    predictions = cv_model.transform(self.test_data)
                    accuracy = evaluator.evaluate(predictions)

                    self.logger.info(f"Modelo {name} - AUC: {accuracy}")
                    self.best_models[name] = cv_model.bestModel

                    # Registrar parámetros y métricas en MLflow
                    mlflow.log_params(cv_model.bestModel.extractParamMap())
                    mlflow.log_metric("auc", accuracy)

                    # Guardar el modelo
                    mlflow.spark.log_model(cv_model.bestModel, f"model_{name}")

            except Exception as e:
                self.logger.error(f"Error al entrenar y evaluar el modelo {name}: {e}")

    def get_best_models(self):
        """
        Retorna los mejores modelos después del entrenamiento y evaluación.
        """
        return self.best_models