from typing import Dict

from scripts.models.model_abstraction import ModelAbstraction

metrics = {
    "non-Decoração": {
        "precision": 0.9640504867616574,
        "recall": 0.9810932785002804,
        "f1-score": 0.9724972206045848,
        "support": 37447,
    },
    "Decoração": {
        "precision": 0.93945613134941,
        "recall": 0.889122693428294,
        "f1-score": 0.9135966735966736,
        "support": 12356,
    },
    "accuracy": 0.9582756058871955,
    "macro avg": {
        "precision": 0.9517533090555337,
        "recall": 0.9351079859642872,
        "f1-score": 0.9430469471006292,
        "support": 49803,
    },
    "weighted avg": {
        "precision": 0.9579486885672971,
        "recall": 0.9582756058871955,
        "f1-score": 0.9578841418978855,
        "support": 49803,
    },
}


def test_flat_dict():
    report: Dict = ModelAbstraction._flat_dict(metrics)
    assert report["accuracy"] == 0.9582756058871955
    assert report["Decoração-precision"] == 0.93945613134941

    assert True
