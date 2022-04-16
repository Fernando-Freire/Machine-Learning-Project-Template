from typing import List

import lightgbm as lgb

from scripts.core.domain import PreProcessedData
from scripts.ports.training_model import TrainingModel


class LightGBM(TrainingModel):
    def __init__(self):
        self.lgb_clf = lgb.LGBMClassifier(random_state=0)

    def predict(self, input_data: PreProcessedData) -> List:
        return self.lgb_clf.predict(input_data.feature_matrix)

    def build(self, train_data: PreProcessedData):
        self.lgb_clf.fit(train_data.feature_matrix, train_data.target)

    def export(self):
        return self.lgb_clf
