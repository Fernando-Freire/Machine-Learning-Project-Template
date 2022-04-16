from typing import List

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from scripts.core.domain import PreProcessedData
from scripts.ports.training_model import TrainingModel


class ScikitModel(TrainingModel):
    def __init__(self, model_type: str):
        if model_type == "DecisionTree":
            self.pipe = Pipeline([("dtc", DecisionTreeClassifier(random_state=0))])
        elif model_type == "ExtraTreeClassifier":
            self.pipe = Pipeline([("etc", ExtraTreeClassifier(random_state=0))])

    def build(self, train_data: PreProcessedData):
        self.pipe.fit(train_data.feature_matrix, train_data.target)

    def predict(self, input_data: PreProcessedData) -> List:
        return self.pipe.predict(input_data.feature_matrix)

    def export(self):
        return self.pipe
