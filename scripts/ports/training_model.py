from abc import ABC, abstractmethod
from typing import List

from scripts.core.domain import PreProcessedData


class TrainingModel(ABC):
    @abstractmethod
    def predict(self, input_data: PreProcessedData) -> List:
        raise NotImplementedError

    @abstractmethod
    def build(self, train_data: PreProcessedData):
        raise NotImplementedError

    @abstractmethod
    def export(self):
        raise NotImplementedError
