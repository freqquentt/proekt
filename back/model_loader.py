import re
from torch.optim.lr_scheduler import StepLR

from models.tabnet import CustomTabNetClassifier
from models.ft_transformer import CustomFTTransformerClassifier
from models.tabtransformer import CustomTabTransformerClassifier
from models.xgb_classifier import CustomXGBClassifier
from models.random_forest import CustomRandomForestClassifier
import numpy as np
import pandas as pd

from typing import Any, List, Tuple, Optional
import torch


TEST_MODELS_DIR = './saved_models'


def get_train_model(
    model_name: str,
    df_categories: List[int],
    df_num_continuous: int,
    y: np.ndarray
) -> Any:

    model: Optional[Any] = None
    if model_name == 'TabTransformer':
        model = CustomTabTransformerClassifier(
            categories=df_categories,
            num_continuous=df_num_continuous,
            dim=32,
            depth=8,
            heads=16,
            dim_head=32,
            attn_dropout=0.1,
            ff_dropout=0.1,
            dim_out=len(np.unique(y))
        )

    elif model_name == 'FT-Transformer':
        model = CustomFTTransformerClassifier(
            categories=df_categories,
            num_continuous=df_num_continuous,
            dim=32,
            depth=6,
            heads=8,
            dim_head=16,
            attn_dropout=0.1,
            ff_dropout=0.1,
            dim_out=len(np.unique(y))
        )

    elif model_name == 'TabNet':
        model = CustomTabNetClassifier(
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=0.3),
            scheduler_params={'step_size': 4, 'gamma': 0.5},
            scheduler_fn=StepLR,
            n_d=32,
            n_a=32,
        )

    elif model_name == 'XGBClassifier':
        model = CustomXGBClassifier(
            n_estimators=400,
            max_depth=10,
            learning_rate=0.001,
            gamma=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            colsample_bylevel=0.8,
            reg_alpha=0.01,
            random_state=42,
            eval_metric='logloss'
        )

    elif model_name == 'RandomForestClassifier':
        model = CustomRandomForestClassifier(
            random_state=42,
            n_estimators = 500,
            max_depth = 10,
            min_samples_split = 3,
            min_samples_leaf = 3
        )

    return model


def get_test_model(
    model_path: str,
    df_categories,
    df_num_continuous,
    y

) -> Any:

    model: Optional[Any] = None

    pattern = r"(TabNet|TabTransformer|FT-Transformer|XGBClassifier|RandomForestClassifier)"
    match = re.search(pattern, model_path)
    model_type = match.group(1)

    if model_type == 'TabTransformer':
        model = CustomTabTransformerClassifier.load(
            model_path,
            categories=df_categories,
            num_continuous=df_num_continuous,
            dim=32,
            depth=8,
            heads=16,
            dim_head=32,
            attn_dropout=0.1,
            ff_dropout=0.1,
            dim_out=len(np.unique(y))
        )

    elif model_type == 'FT-Transformer':
        model = CustomFTTransformerClassifier.load(
            model_path,
            categories=df_categories,
            num_continuous=df_num_continuous,
            dim=32,
            depth=6,
            heads=8,
            dim_head=16,
            attn_dropout=0.1,
            ff_dropout=0.1,
            dim_out=len(np.unique(y))
        )

    elif model_type == 'TabNet':
        model = CustomTabNetClassifier.load(model_path)

    elif model_type == 'XGBClassifier':
        model = CustomXGBClassifier.load(model_path)

    elif model_type == 'RandomForestClassifier':
        model = CustomRandomForestClassifier.load(model_path)

    return model


def analyze_dataset(df: pd.DataFrame) -> Tuple[List[str], List[str], List[int]]:

    cat_columns: List[str] = []
    cont_columns: List[str] = []

    for col in df.columns:
        if df[col].nunique() < 15:
            cat_columns.append(col)
        else:
            cont_columns.append(col)

    categories = [df[col].nunique() for col in cat_columns]
    return cont_columns, cat_columns, categories



