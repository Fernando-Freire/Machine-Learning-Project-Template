from abc import ABC, abstractmethod
from typing import Dict

from scripts.ports.training_model import TrainingModel
from scripts.ports.word_embedding_gateway import WordEmbeddingModel


class ModelVersioning(ABC):
    @staticmethod
    @abstractmethod
    def export_training_model(
        model_algorithm: str,
        category: str,
        training_model: TrainingModel,
        metrics: Dict,
    ):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def export_word_embedding_model(
        word_embedding: str, category: str, word_embedding_model: WordEmbeddingModel
    ):
        raise NotImplementedError
