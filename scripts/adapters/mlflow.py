from typing import Dict

import mlflow

from scripts.ports.model_versioning_gateway import ModelVersioning
from scripts.ports.training_model import TrainingModel
from scripts.ports.word_embedding_gateway import WordEmbeddingModel


class MLFlow(ModelVersioning):
    def __init__(self, mlflow_uri: str, word_embedding: str, model_algorithm: str):
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name=word_embedding + "-" + model_algorithm)

    @staticmethod
    def export_word_embedding_model(
        word_embedding: str, category: str, word_embedding_model: WordEmbeddingModel
    ):
        with mlflow.start_run(run_name=word_embedding + "-" + category):
            if word_embedding == "bow" or word_embedding == "tfidf":
                mlflow.sklearn.log_model(
                    sk_model=word_embedding_model.get_model(),
                    artifact_path="artifacts",
                    registered_model_name=word_embedding + "-" + category,
                )
            elif (
                word_embedding == "Word2Vec"
                or word_embedding == "Doc2Vec"
                or word_embedding == "FastText"
            ):
                mlflow.pyfunc.log_model(
                    artifact_path="artifacts",
                    python_model=word_embedding_model.get_model(),
                    registered_model_name=word_embedding + "-" + category,
                )

    @staticmethod
    def export_training_model(
        model_algorithm: str,
        category: str,
        training_model: TrainingModel,
        metrics: Dict,
    ):
        with mlflow.start_run(run_name=model_algorithm + "-" + category):
            mlflow.log_metrics(metrics=metrics)
            if (
                model_algorithm == "DecisionTree"
                or model_algorithm == "ExtraTreeClassifier"
            ):
                mlflow.sklearn.log_model(
                    sk_model=training_model.export(),
                    artifact_path="artifacts",
                    registered_model_name=model_algorithm + "-" + category,
                )
            elif model_algorithm == "lightgbm":
                mlflow.lightgbm.log_model(
                    lgb_model=training_model.export(),
                    artifact_path="artifacts",
                    registered_model_name=model_algorithm + "-" + category,
                )
