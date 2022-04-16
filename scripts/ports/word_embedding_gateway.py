from abc import ABC, abstractmethod

from scripts.core.domain import DataXY, PreProcessedData


class WordEmbeddingModel(ABC):
    @abstractmethod
    def fit_and_transform(self, train_data: DataXY) -> PreProcessedData:
        raise NotImplementedError

    @abstractmethod
    def transform(self, test_data: DataXY) -> PreProcessedData:
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        raise NotImplementedError
