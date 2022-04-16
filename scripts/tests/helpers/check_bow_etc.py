from pydantic import BaseSettings
from scripts.core.config import MainConfig

from scripts.core.training import run


config = MainConfig()

metrics = run(
    config, word_embedding="bow", model_algorithm="ExtraTreeClassifier", test_size=0.0
)


for metric in metrics.keys():
    print("métricas para a categoria: " + metric)
    print(metrics[metric])
    assert metrics[metric]["accuracy"] > 0.9
