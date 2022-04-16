from typing import Dict

from pydantic import BaseSettings
from scripts.core.config import MainConfig

from scripts.core.training import run


config = MainConfig()

metrics: Dict = run(
    config, word_embedding="bow", model_algorithm="DecisionTree", test_size=0.0
)

# print(metrics)
# print(type(metrics))
# print(metrics.keys())
for metric in metrics.keys():
    print("métricas para a categoria: " + metric)
    print(metrics[metric])
    assert metrics[metric]["accuracy"] > 0.9
