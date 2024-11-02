import numpy as np
from sklearn.linear_model import LogisticRegression
from src.entities import TrainingParams


def train_model(features: np.array, target: np.array, training_params: TrainingParams):
    if training_params.model_type == "LogisticRegression":
        model = LogisticRegression(random_state=training_params.random_state)
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model
