from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import classification_report

from scripts.core.domain import PreProcessedData
from scripts.models.training_models.lightgbm import LightGBM
from scripts.models.training_models.scikit_model import ScikitModel
from scripts.models.word_embeddings.gensim import Gensim
from scripts.models.word_embeddings.scikit_embedding import ScikitEmbedding
from scripts.ports.training_model import TrainingModel
from scripts.ports.word_embedding_gateway import WordEmbeddingModel


class ModelAbstraction:
    @staticmethod
    def define_models(
        word_embedding: str, model_algorithm: str
    ) -> Tuple[WordEmbeddingModel, TrainingModel]:
        word_embedding_model: WordEmbeddingModel
        model: TrainingModel
        if word_embedding == "bow" or word_embedding == "tfidf":
            word_embedding_model = ScikitEmbedding(word_model_type=word_embedding)
        elif (
            word_embedding == "Word2Vec"
            or word_embedding == "Doc2Vec"
            or word_embedding == "FastText"
        ):
            word_embedding_model = Gensim(word_model_type=word_embedding)

        if (
            model_algorithm == "DecisionTree"
            or model_algorithm == "ExtraTreeClassifier"
        ):
            model = ScikitModel(model_type=model_algorithm)
        elif model_algorithm == "lightgbm":
            model = LightGBM()

        return word_embedding_model, model

    @staticmethod
    def get_report(
        test_predictions, test_data: PreProcessedData, category_name: str
    ) -> Dict:
        report = classification_report(
            test_predictions,
            test_data.target,
            output_dict=True,
            target_names=["non-category", "is-cateogory"],
        )

        return ModelAbstraction._flat_dict(report=report)

    @staticmethod
    def _flat_dict(report: Dict) -> Dict:

        [flat_dict] = pd.json_normalize(report, sep="_").to_dict(orient="records")

        return flat_dict
