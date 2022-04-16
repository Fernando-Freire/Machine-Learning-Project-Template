from pydantic import BaseSettings

from scripts.core.training import run
from scripts.core.config import MainConfig


class RunConfig(BaseSettings):
    test_size: float
    model_algorithim: str
    word_embedding: str


if __name__ == "__main__":

    main_config = MainConfig()
    run_config = RunConfig()

    run(
        main_config,
        word_embedding=run_config.word_embedding,
        model_algorithm=run_config.model_algorithim,
        test_size=run_config.test_size,
    )
