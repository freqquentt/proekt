from sklearn.preprocessing import LabelEncoder
from .base_model import CustomBaseModel
from tab_transformer_pytorch import TabTransformer
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Any, List, Tuple, Optional
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


class CustomTabTransformerClassifier(CustomBaseModel):

    def __init__(self, **kwargs: Any) -> None:

        super().__init__()
        self.model: TabTransformer = TabTransformer(**kwargs)
        self.le: Optional[LabelEncoder] = None

    def fit(
        self,
        target_column: np.ndarray,
        df: pd.DataFrame,
        cat_columns: Optional[List[str]] = None,
        cont_columns: Optional[List[str]] = None,
        batch_size: int = 4096,
    ) -> Tuple[List[float], List[float]]:

        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(target_column)
        print(f"[INFO] Target vector shape: {y_encoded.shape}")

        train_losses: List[float] = []
        valid_losses: List[float] = []
        LR = 1e-3
        MAX_EPOCH = 5

        if cat_columns is None or cont_columns is None:
            raise ValueError("cat_columns and cont_columns must be provided")

        X_cat = df[cat_columns].values
        X_cont = df[cont_columns].values.astype("float32")
        print(f"[INFO] Categorical data shape: {X_cat.shape}")
        print(f"[INFO] Continuous data shape: {X_cont.shape}")

        for i, col in enumerate(cat_columns):
            unique_vals = np.unique(X_cat[:, i])
            print(f"[INFO] Category '{col}' - unique values: {unique_vals}")

        indices = np.arange(len(y_encoded))
        train_idx, valid_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=y_encoded
        )
        print("[INFO] Data split into training and validation sets")

        X_cat_train = X_cat[train_idx]
        X_cat_valid = X_cat[valid_idx]
        X_cont_train = X_cont[train_idx]
        X_cont_valid = X_cont[valid_idx]
        y_train = y_encoded[train_idx]
        y_valid = y_encoded[valid_idx]

        X_cat_train = torch.tensor(X_cat_train, dtype=torch.long)
        X_cat_valid = torch.tensor(X_cat_valid, dtype=torch.long)
        X_cont_train = torch.tensor(X_cont_train, dtype=torch.float32)
        X_cont_valid = torch.tensor(X_cont_valid, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_valid = torch.tensor(y_valid, dtype=torch.long)
        print("[INFO] Data converted to tensors")

        train_dataset = TensorDataset(X_cat_train, X_cont_train, y_train)
        valid_dataset = TensorDataset(X_cat_valid, X_cont_valid, y_valid)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
        criterion = nn.CrossEntropyLoss()

        device = torch.device("cpu")
        self.model.to(device)

        for epoch in range(MAX_EPOCH):
            print(f"[INFO] Starting epoch {epoch + 1}/{MAX_EPOCH}")
            self.model.train()
            running_loss = 0.0

            for X_cat_batch, X_cont_batch, y_batch in train_loader:
                X_cat_batch = X_cat_batch.to(device)
                X_cont_batch = X_cont_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = self.model(X_cat_batch, X_cont_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X_cat_batch.size(0)

            scheduler.step()

            train_loss_epoch = running_loss / len(train_dataset)

            self.model.eval()
            valid_loss_epoch = 0.0
            with torch.no_grad():
                for X_cat_batch, X_cont_batch, y_batch in valid_loader:
                    X_cat_batch = X_cat_batch.to(device)
                    X_cont_batch = X_cont_batch.to(device)
                    y_batch = y_batch.to(device)
                    outputs = self.model(X_cat_batch, X_cont_batch)
                    loss = criterion(outputs, y_batch)
                    valid_loss_epoch += loss.item() * X_cat_batch.size(0)

            avg_train_loss = train_loss_epoch
            avg_valid_loss = valid_loss_epoch / len(valid_dataset)
            train_losses.append(avg_train_loss)
            valid_losses.append(avg_valid_loss)

            print(f"[INFO] Epoch {epoch + 1}/{MAX_EPOCH} - Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

            if epoch >= 2:
                last_three_train = train_losses[-4:]
                if max(last_three_train) - min(last_three_train) <= 0.0001:
                    print("[INFO] Early stopping: Train loss changes less than 0.001 for 4 consecutive epochs.")
                    break

                last_three_valid = valid_losses[-4:]
                if all(v > t for v, t in zip(last_three_valid, train_losses[-4:])):
                    print("[INFO] Early stopping: Validation loss is higher than train loss for 4 consecutive epochs.")
                    break

        print("[INFO] Training completed")
        return train_losses, valid_losses

    def predict(self, df: pd.DataFrame, cat_columns: List[str] = None, cont_columns: List[str] = None):

        self.model.eval()
        with torch.no_grad():
            X_cat = torch.tensor(df[cat_columns].values, dtype=torch.long)
            X_cont = torch.tensor(df[cont_columns].values.astype("float32"))
            outputs = self.model(X_cat, X_cont)
            _, predicted = torch.max(outputs, dim=1)
        return predicted

    def save(self, model_path: str) -> None:

        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "classes": self.le.classes_ if self.le is not None else None,
        }
        torch.save(save_dict, model_path)
        print(f"[INFO] Model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str, **kwargs: Any) -> 'CustomTabTransformerClassifier':

        instance = cls(**kwargs)
        checkpoint = torch.load(model_path, weights_only=False)
        instance.model.load_state_dict(checkpoint["model_state_dict"])

        if "classes" in checkpoint and checkpoint["classes"] is not None:
            instance.le = LabelEncoder()
            instance.le.classes_ = checkpoint["classes"]
        instance.model.eval()
        print(f"[INFO] Model loaded from {model_path}")
        return instance
