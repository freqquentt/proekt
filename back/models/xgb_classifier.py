import pickle
from sklearn.preprocessing import LabelEncoder
from .base_model import CustomBaseModel
from typing import Any, Optional, List, Tuple
from xgboost import XGBClassifier
import numpy as np
import pandas as pd


class CustomXGBClassifier(CustomBaseModel):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.model = XGBClassifier(**kwargs)
        print("[INFO] XGBClassifier model initialized.")

    def fit(
            self,
            target_column: np.ndarray,
            df: pd.DataFrame,
            cat_columns: Optional[List[str]] = None,
            cont_columns: Optional[List[str]] = None,
            batch_size: int = 4096,
    ):
        print("[INFO] Starting model training.")

        le = LabelEncoder()
        y_encoded = le.fit_transform(target_column)

        X = df.values

        self.model.fit(X, y_encoded)

        print("[INFO] Model training completed.")


    def predict(self, df: pd.DataFrame, cat_columns: List[str] = None, cont_columns: List[str] = None):

        print("[INFO] Starting prediction.")

        X = df.values
        predictions = self.model.predict(X)

        print("[INFO] Prediction completed.")
        return predictions

    def save(self, model_path: str):

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[INFO] Model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str, **kwargs):

        print(f"[INFO] Loading model from {model_path}")
        instance = cls(**kwargs)
        with open(model_path, 'rb') as f:
            instance.model = pickle.load(f)
        print("[INFO] Model successfully loaded.")
        return instance



