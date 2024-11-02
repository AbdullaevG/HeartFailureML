from typing import Dict
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def model_predict(features: np.array, model: LogisticRegression) -> np.array:
    predict = model.predict(features)
    return predict


def evaluate_model(predict: np.array, target: np.array) -> Dict:
    accuracy = accuracy_score(predict, target)
    f1 = f1_score(predict, target).item()
    roc_auc = roc_auc_score(predict, target).item()
    return {"accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc}


def save_model(model: LogisticRegression, model_save_path: str):
    with open(model_save_path, "wb") as save_file:
        pickle.dump(model, save_file)
