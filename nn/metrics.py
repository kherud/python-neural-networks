import numpy as np
from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def evaluate(self, prediction, truth):
        pass


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.accuracy = 0

    def __str__(self):
        return "acc = {:.4f}".format(self.accuracy)

    def evaluate(self, prediction, truth):
        self.accuracy = np.mean(prediction == truth)


class F1Score(Metric):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def __str__(self):
        if self.verbose:
            return "f1 = {:.4f}; pr = {:.4f}; re = {:.4f}".format(self.f1, self.precision, self.recall)
        else:
            return "f1 = {:.4f}".format(np.mean(self.f1))

    def evaluate(self, prediction, truth):
        confusion_matrix = np.zeros(shape=(10, 10)).astype(int)
        for index, (prediction, target) in enumerate(zip(prediction, truth)):
            confusion_matrix[target, prediction] += 1
        self.precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        self.recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
