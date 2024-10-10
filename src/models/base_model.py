from abc import ABC, abstractmethod
from typing import Dict, Any

class Model(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        pass