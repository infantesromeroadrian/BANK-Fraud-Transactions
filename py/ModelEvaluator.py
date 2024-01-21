from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
import logging



class ModelEvaluator:
    def __init__(self, predictions, labelCol="label", predictionCol="prediction", probabilityCol="probability"):
        self.predictions = predictions
        self.labelCol = labelCol
        self.predictionCol = predictionCol
        self.probabilityCol = probabilityCol

    def evaluate_accuracy(self):
        evaluator = MulticlassClassificationEvaluator(labelCol=self.labelCol, predictionCol=self.predictionCol, metricName="accuracy")
        return evaluator.evaluate(self.predictions)

    def evaluate_precision(self):
        evaluator = MulticlassClassificationEvaluator(labelCol=self.labelCol, predictionCol=self.predictionCol, metricName="weightedPrecision")
        return evaluator.evaluate(self.predictions)

    def evaluate_recall(self):
        evaluator = MulticlassClassificationEvaluator(labelCol=self.labelCol, predictionCol=self.predictionCol, metricName="weightedRecall")
        return evaluator.evaluate(self.predictions)

    def evaluate_f1_score(self):
        evaluator = MulticlassClassificationEvaluator(labelCol=self.labelCol, predictionCol=self.predictionCol, metricName="f1")
        return evaluator.evaluate(self.predictions)

    def evaluate_auc_roc(self):
        evaluator = BinaryClassificationEvaluator(labelCol=self.labelCol, rawPredictionCol=self.probabilityCol, metricName="areaUnderROC")
        return evaluator.evaluate(self.predictions)

    def plot_confusion_matrix(self):
        y_true = self.predictions.select(self.labelCol).toPandas()
        y_pred = self.predictions.select(self.predictionCol).toPandas()
        cm = pd.crosstab(y_true[self.labelCol], y_pred[self.predictionCol], rownames=['Actual'], colnames=['Predicted'])

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
