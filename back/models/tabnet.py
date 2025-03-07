from .base_model import CustomBaseModel
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import LabelEncoder



class CustomTabNetClassifier(CustomBaseModel):
    def __init__(self, **kwargs):

        super().__init__()
        self.model = TabNetClassifier(**kwargs)
        print("[INFO] TabNetClassifier model initialized.")

    def fit(
        self,
        target_column: np.ndarray,
        df: pd.DataFrame,
        cat_columns: Optional[List[str]] = None,
        cont_columns: Optional[List[str]] = None,
    ):

        print("[INFO] Starting model training.")

        X = df.values
        y = target_column

        le = LabelEncoder()
        y_encoded = le.fit_transform(target_column)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        print(f"[INFO] Training dataset size: {X_train.shape[0]} records")
        print(f"[INFO] Validation dataset size: {X_valid.shape[0]} records")

        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=["valid"],
            max_epochs=5,
            patience=10,
            batch_size=512,
            virtual_batch_size=256,
        )

        print("[INFO] Model training completed.")

    def predict(self, df: pd.DataFrame, cat_columns: List[str] = None, cont_columns: List[str] = None):

        print("[INFO] Starting prediction.")

        X = df.values
        predictions = self.model.predict(X)

        print("[INFO] Prediction completed.")
        return predictions

    def save(self, model_path: str):

        self.model.save_model(model_path)
        print(f"[INFO] Model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str, **kwargs):

        print(f"[INFO] Loading model from {model_path}")

        instance = cls(**kwargs)
        instance.model.load_model(model_path)

        print("[INFO] Model successfully loaded.")
        return instance
