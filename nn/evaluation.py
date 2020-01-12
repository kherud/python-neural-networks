from abc import ABC


class Metric(ABC):
    def __init__(self):
        pass

    def evaluate(self, y):
        pass