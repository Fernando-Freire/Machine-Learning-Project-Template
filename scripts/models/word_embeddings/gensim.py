from typing import List

import mlflow.pyfunc
import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from numpy import ndarray

from scripts.core.domain import DataXY, PreProcessedData
from scripts.ports.word_embedding_gateway import WordEmbeddingModel


class Gensim(WordEmbeddingModel, mlflow.pyfunc.PythonModel):
    def __init__(self, word_model_type: str, dimensions=100):
        self.dimensions = dimensions
        self.word_model_type = word_model_type
        self.model: Word2Vec
        if word_model_type == "Word2Vec":
            self.model = Word2Vec(min_count=1)

        elif word_model_type == "FastText":
            self.model = FastText(min_count=1)

        elif word_model_type == "Doc2Vec":
            self.model = Doc2Vec(min_count=1)

    @staticmethod
    def _pre_process_data(data: pd.Series) -> pd.Series:
        return pd.Series([simple_preprocess(line, deacc=True) for line in data])

    @staticmethod
    def _get_features(
        prepared_train_data: pd.Series, word_model_type: str
    ) -> pd.Series:
        if word_model_type == "Doc2Vec":
            return pd.Series(
                [TaggedDocument(doc, [i]) for i, doc in enumerate(prepared_train_data)]
            )
        else:
            return prepared_train_data

    @staticmethod
    def _get_feature_matrix(
        features: pd.Series, word_model_type: str, model, dimensions
    ):
        features_applied: pd.Series
        if word_model_type == "Doc2Vec":
            features_applied = features.apply(
                lambda line: Gensim._codify(
                    tokens=line.words,
                    word_model_type=word_model_type,
                    model=model,
                    dimensions=dimensions,
                )
            )
        else:
            features_applied = features.apply(
                lambda line: Gensim._codify(
                    tokens=line,
                    word_model_type=word_model_type,
                    model=model,
                    dimensions=dimensions,
                )
            )
        features_matrix = np.stack(features_applied.tolist(), axis=0)
        return features_matrix

    @staticmethod
    def _codify(tokens: List[str], word_model_type: str, model, dimensions) -> ndarray:
        if word_model_type == "FastText":
            np_array = np.mean([model.wv.get_vector(token) for token in tokens], axis=0)
        elif word_model_type == "Doc2Vec":
            np_array = model.infer_vector(doc_words=tokens, epochs=10)
        else:
            np_array = np.mean(
                [model.wv.get_vector(token) for token in tokens if token in model.wv],
                axis=0,
            )

        if np_array.size == dimensions:
            return np_array
        else:
            return np.zeros([dimensions])

    def fit_and_transform(self, train_data: DataXY) -> PreProcessedData:
        prepared_train_data = self._pre_process_data(train_data.features)
        features = self._get_features(
            prepared_train_data=prepared_train_data,
            word_model_type=self.word_model_type,
        )
        self.model.build_vocab(features)
        self.model.train(
            features,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )
        # save local file of model
        # self.model.save()

        # finish saving model
        features_matrix = self._get_feature_matrix(
            features=features,
            word_model_type=self.word_model_type,
            model=self.model,
            dimensions=self.dimensions,
        )
        return PreProcessedData(
            feature_matrix=features_matrix, target=train_data.target
        )

    def transform(self, test_data: DataXY) -> PreProcessedData:
        prepared_train_data = self._pre_process_data(test_data.features)
        features = self._get_features(
            prepared_train_data=prepared_train_data,
            word_model_type=self.word_model_type,
        )
        features_matrix = self._get_feature_matrix(
            features=features,
            word_model_type=self.word_model_type,
            model=self.model,
            dimensions=self.dimensions,
        )
        return PreProcessedData(feature_matrix=features_matrix, target=test_data.target)

    def get_model(self):
        return self

    def _export_model(self):
        return self.model

    def predict(self, context, model_input):
        prepared_model_input = self._pre_process_data(model_input)
        features = self._get_features(
            prepared_train_data=prepared_model_input,
            word_model_type=self.word_model_type,
        )
        return self.self._get_feature_matrix(
            features=features,
            word_model_type=self.word_model_type,
            model=self.model,
            dimensions=self.dimensions,
        )
