import numpy as np


def accuracy(prediction: np.array, truth: np.array) -> float:
    return np.mean(np.array(prediction) == np.array(truth)).item()


def f1_score_mean(prediction: np.array, truth: np.array) -> float:
    return np.mean(f1_score(prediction, truth)).item()


def f1_score(prediction: np.array, truth: np.array) -> np.array:
    confusion_matrix = np.zeros(shape=(10, 10)).astype(int)
    for index, (prediction, target) in enumerate(zip(prediction, truth)):
        confusion_matrix[target, prediction] += 1
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    return 2 * precision * recall / (precision + recall)
