from abc import ABC, abstractmethod
from typing import List


class CustomBaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, target_column, df, cat_columns=None, cont_columns=None):
        pass

    @abstractmethod
    def predict(self, df, cat_columns: List[str] = None, cont_columns: List[str] = None):
        pass

    @abstractmethod
    def save(self, model_path):
        pass

    @abstractmethod
    def load(self, model_path, **kwargs):
        pass
