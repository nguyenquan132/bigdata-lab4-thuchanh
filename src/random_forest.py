from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class RandomForestModel:
    def __init__(self, numTrees=100, maxDepth=5):
        self.model = None
        self.numTrees = numTrees
        self.maxDepth = maxDepth

    def train(self, df: DataFrame):
        if df.count() == 0:
            return [], 0.0, 0.0, 0.0, 0.0

        if self.model is None:
            self.model = RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                numTrees=self.numTrees,
                maxDepth=self.maxDepth,
                seed=42
            )
        model = self.model.fit(df)
        predictions = model.transform(df)
        total = predictions.count()
        correct = predictions.filter(predictions.label == predictions.prediction).count()
        accuracy = correct / total if total > 0 else 0.0

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

        pred_list = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
        return pred_list, accuracy, precision, recall, f1

    def predict(self, df: DataFrame, raw_model=None):
        if df.count() == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        predictions = self.model.transform(df)
        total = predictions.count()
        correct = predictions.filter(predictions.label == predictions.prediction).count()
        accuracy = correct / total if total > 0 else 0.0

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

        # Simplified loss and confusion matrix
        loss = 1.0 - accuracy
        cm = 0.0  # Placeholder; implement if needed
        return accuracy, loss, precision, recall, f1, cm