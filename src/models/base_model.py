from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        pass