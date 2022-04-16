from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from scripts.core.domain import DataXY, PreProcessedData
from scripts.ports.word_embedding_gateway import WordEmbeddingModel


class ScikitEmbedding(WordEmbeddingModel):
    def __init__(self, word_model_type: str):
        if word_model_type == "bow":
            self.pipe = Pipeline([("count", CountVectorizer())])

        elif word_model_type == "tfidf":
            self.pipe = Pipeline(
                [("count", CountVectorizer()), ("tfid", TfidfTransformer())]
            )

    def fit_and_transform(self, train_data: DataXY) -> PreProcessedData:
        self.pipe.fit(train_data.features)
        return self.transform(train_data)

    def transform(self, test_data: DataXY) -> PreProcessedData:
        feature_matrix = self.pipe.transform(test_data.features)
        return PreProcessedData(
            feature_matrix=feature_matrix.toarray(),
            target=test_data.target,
        )

    def get_model(self):
        return self.pipe
