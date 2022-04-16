from typing import Dict, List

from pydantic import BaseSettings

from scripts.adapters.mlflow import MLFlow
from scripts.adapters.pandas import Pandas
from scripts.core.domain import DataXY, PreProcessedData
from scripts.models.model_abstraction import ModelAbstraction
from scripts.ports.model_versioning_gateway import ModelVersioning
from scripts.ports.training_model import TrainingModel
from scripts.ports.word_embedding_gateway import WordEmbeddingModel
from scripts.core.config import MainConfig


def run(
    config: MainConfig, word_embedding: str, model_algorithm: str, test_size: float
) -> Dict:
    word_embedding_model: WordEmbeddingModel
    training_model: TrainingModel

    word_embedding_model, training_model = ModelAbstraction.define_models(
        word_embedding, model_algorithm
    )

    model_versioner: ModelVersioning = MLFlow(
        mlflow_uri=config.mlflow_uri,
        word_embedding=word_embedding,
        model_algorithm=model_algorithm,
    )

    storage_options = {
        "key": config.aws_access_key_id,
        "secret": config.aws_secret_access_key,
        "client_kwargs": {"endpoint_url": config.s3_uri},
    }
    filepath = config.train_data_s3_path + config.train_data_file_name

    metrics_dict: Dict = {}

    for category in Pandas.get_classes(storage_options, filepath):

        train_data: DataXY
        test_data: DataXY

        train_data, test_data = Pandas.get_dataset_splited(
            storage_options, filepath, class_name=category, test_size=test_size
        )

        pre_processed_data: PreProcessedData = word_embedding_model.fit_and_transform(
            train_data=train_data
        )
        training_model.build(train_data=pre_processed_data)

        pre_processed_test_data: PreProcessedData = word_embedding_model.transform(test_data=test_data)
        test_predictions: List = training_model.predict(pre_processed_test_data)

        training_report = ModelAbstraction.get_report(
            test_predictions, pre_processed_data, category_name=category
        )
        metrics_dict[category] = training_report
        # save Word embedding model

        model_versioner.export_word_embedding_model(
            word_embedding=word_embedding,
            category=category,
            word_embedding_model=word_embedding_model,
        )

        model_versioner.export_training_model(
            model_algorithm=model_algorithm,
            category=category,
            training_model=training_model,
            metrics=training_report,
        )

    return metrics_dict
