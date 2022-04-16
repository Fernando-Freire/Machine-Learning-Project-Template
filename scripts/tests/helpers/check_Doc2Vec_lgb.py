from pydantic import BaseSettings
from scripts.core.config import MainConfig

from scripts.core.training import run


config = MainConfig()

metrics = run(
    config, word_embedding="Doc2Vec", model_algorithm="lightgbm", test_size=0.0
)


for metric in metrics.keys():
    print("métricas para a categoria: " + metric)
    print(metrics[metric]["accuracy"])
    # Doc2Vec results in a relative lower accuracy
    # for tagged documents with few words or small datasets
    assert metrics[metric]["accuracy"] > 0.75
