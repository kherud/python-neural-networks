import numpy as np


def accuracy(prediction: np.array, truth: np.array) -> float:
    return np.mean(np.array(prediction) == np.array(truth)).item()


def f1_score_mean(prediction: np.array, truth: np.array) -> float:
    return np.mean(f1_score(prediction, truth)).item()


def f1_score(prediction: np.array, truth: np.array) -> np.array:
    conf_matrix = confusion_matrix(prediction, truth)
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    return 2 * precision * recall / (precision + recall)


def confusion_matrix(prediction: np.array, truth: np.array) -> np.array:
    n_classes = np.unique(truth).shape[0]
    conf_matrix = np.zeros(shape=(n_classes, n_classes)).astype(np.int)
    for index, (prediction, target) in enumerate(zip(prediction, truth)):
        conf_matrix[target, prediction] += 1
    return conf_matrix
