import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Any, List, Tuple, Optional


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.label_encoders = {}
        self.columns = None

    def fit(self, X, y=None):

        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
        self.columns = X.columns
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self

    def transform(self, X):

        if not hasattr(X, "columns"):
            X = pd.DataFrame(X, columns=self.columns)
        X_transformed = X.copy()
        for col in self.columns:
            if col in self.label_encoders:
                X_transformed[col] = self.label_encoders[col].transform(X[col])
        return X_transformed.values


class DataCleaner:

    def __init__(self, target_column: Optional[str] = None) -> None:

        self.target_column: Optional[str] = target_column
        self.pipeline: Optional[ColumnTransformer] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []

    def _identify_features(self, df: pd.DataFrame):

        if self.target_column in df.columns:
            df_features = df.drop(columns=[self.target_column])
        else:
            df_features = df.copy()

        df_features = df_features.loc[:, ~df_features.columns.str.contains('id', case=False)]

        self.numeric_features = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.text_features = []
        self.categorical_features = []

        for col in df_features.select_dtypes(include=['object']).columns:
                self.categorical_features.append(col)


    def _build_pipeline(self):

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('label_enc', MultiColumnLabelEncoder())
        ])
        self.pipeline = ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.numeric_features),
            ('cat', categorical_transformer, self.categorical_features),
        ], sparse_threshold=0.0)


    def fit(self, df: pd.DataFrame):

        self._identify_features(df)
        self._build_pipeline()
        if self.target_column in df.columns:
            df_features = df.drop(columns=[self.target_column])
        else:
            df_features = df.copy()
        self.pipeline.fit(df_features)
        return self

    def transform(self, df: pd.DataFrame):

        if self.target_column in df.columns:
            df_features = df.drop(columns=[self.target_column])
        else:
            df_features = df.copy()
        return self.pipeline.transform(df_features)

    def fit_transform(self, df: pd.DataFrame):

        self.fit(df)
        return self.transform(df)