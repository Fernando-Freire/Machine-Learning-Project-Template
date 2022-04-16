import mlflow
from pydantic import BaseSettings

from scripts.adapters.mlflow import MLFlow
from scripts.models.training_models.lightgbm import LightGBM
from scripts.models.word_embeddings.scikit_embedding import ScikitEmbedding
from scripts.ports.training_model import TrainingModel
from scripts.ports.word_embedding_gateway import WordEmbeddingModel


class TestMLFlowConfig(BaseSettings):
    mlflow_tracking_uri: str


def test_export_word_embedding_model():
    config = TestMLFlowConfig()
    mlflow_obj = MLFlow(
        mlflow_uri=config.mlflow_tracking_uri,
        word_embedding="Test_Word_model",
        model_algorithm="Test_Algo_model",
    )

    embedding_model: WordEmbeddingModel = ScikitEmbedding(word_model_type="bow")
    mlflow_obj.export_word_embedding_model(
        word_embedding="Word_model",
        category="fruta",
        word_embedding_model=embedding_model,
    )

    last_run = mlflow.last_active_run()
    assert last_run.info.status == "FINISHED"


def test_export_training_model():
    config = TestMLFlowConfig()
    mlflow_obj = MLFlow(
        mlflow_uri=config.mlflow_tracking_uri,
        word_embedding="Test_Word_model",
        model_algorithm="Test_Algo_model",
    )

    model: TrainingModel = LightGBM()

    mlflow_obj.export_training_model(
        model_algorithm="Algo_model",
        category="fruta",
        training_model=model,
        metrics={"doce": 0.9},
    )

    last_run = mlflow.last_active_run()
    assert last_run.data.metrics["doce"] == 0.9
