# main_script.py

from DataIngestor import DataIngestor
from DataExplorer import DataExplorer
from DataTransformer import DataTransformer
from DataSplitter import Splitter
from ModelTrainer import ModelTrainer
from ModelEvaluator import ModelEvaluator
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
import mlflow.spark
import logging

# Ingesta de datos
file_path = "/path/to/your/data.csv"
data_ingestor = DataIngestor(file_path)
bank_data = data_ingestor.read_data()

# Exploración de datos
data_explorer = DataExplorer(bank_data)
data_explorer.show_info()
data_explorer.show_describe()
data_explorer.show_null_values()

# Transformación de datos
data_transformer = DataTransformer(bank_data)
columns_to_encode = ['type', 'Card Type', 'Exp Type', 'Gender']
transformed_data = data_transformer.one_hot_encode(columns_to_encode)
transformed_data = data_transformer.convert_to_date('Date')
columns_to_scale = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
transformed_data = data_transformer.scale_features(columns_to_scale)

# Preparación y división de datos
model_prep = Splitter(transformed_data)
feature_columns = ['type_indexed', 'type_encoded', 'Card Type_indexed', 'Card Type_encoded', 'Exp Type_indexed', 'Exp Type_encoded', 'Gender_indexed', 'Gender_encoded', 'scaled_features']
label_column = 'isFraud'
prepared_data = model_prep.select_features(feature_columns, label_column)
train_data, test_data = model_prep.split_data(train_ratio=0.7)

# Entrenamiento de modelos
models = {
    "RandomForest": {
        "model": RandomForestClassifier(featuresCol='features', labelCol='label'),
        "param_grid": {
            RandomForestClassifier.numTrees: [10, 20, 30],
            RandomForestClassifier.maxDepth: [5, 10]
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(featuresCol='features', labelCol='label'),
        "param_grid": {
            LogisticRegression.maxIter: [10, 20],
            LogisticRegression.regParam: [0.01, 0.1]
        }
    }
}
trainer = ModelTrainer(train_data, test_data, models)
trainer.fit_and_evaluate()
best_models = trainer.get_best_models()

# Guardando los mejores modelos
for name, model in best_models.items():
    mlflow.spark.save_model(model, f"models/model_{name}")

# Evaluación de modelos
# Suponiendo que usas RandomForest como ejemplo
rf_model = best_models['RandomForest']
predictions = rf_model.transform(test_data)
evaluator = ModelEvaluator(predictions)
print(f"Accuracy: {evaluator.evaluate_accuracy()}")
print(f"F1 Score: {evaluator.evaluate_f1_score()}")
print(f"AUC ROC: {evaluator.evaluate_auc_roc()}")
evaluator.plot_confusion_matrix()
