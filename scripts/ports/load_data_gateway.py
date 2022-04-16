from abc import ABC, abstractmethod
from typing import Dict, List

from scripts.core.domain import DataXY


class LoadDataGateway(ABC):
    @staticmethod
    @abstractmethod
    def get_dataset_splited(
        storage_options: Dict,
        filepath: str,
        class_name: str,
        test_size: float,
        **kwargs
    ) -> tuple[DataXY, DataXY]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_classes(storage_options: Dict, filepath: str, **kwargs) -> List[str]:
        raise NotImplementedError
