import os
import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse
from sklearn.preprocessing import LabelEncoder
from schemas.prediction import TrainPredictionRequest, TestPredictionRequest
from schemas.metrics import ClassificationMetrics
from model_loader import get_train_model, analyze_dataset, get_test_model
from preprocessing.preprocessing_pipeline import DataCleaner
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             cohen_kappa_score, jaccard_score, hamming_loss, accuracy_score)


app = FastAPI()

TRAINED_MODEL_DIR = "./saved_models"


@app.post("/train/")
def train(request: TrainPredictionRequest) -> dict:

    try:
        target_column: str = request.target_column

        csv_buffer = io.StringIO(request.csv_data)
        df = pd.read_csv(csv_buffer, low_memory=False)

        print(df.dtypes)

        X = df.copy().drop(columns=[target_column])
        y  = df[target_column].values

        print("Cleaning and processing data...")
        cleaner = DataCleaner(target_column=target_column)
        processed_X = cleaner.fit_transform(X)
        print("Data cleaning completed.")

        new_columns = cleaner.numeric_features + cleaner.categorical_features
        processed_df = pd.DataFrame(processed_X, columns=new_columns)

        cont_columns, cat_columns, num_categories = analyze_dataset(processed_df)

        model = get_train_model(
            model_name=request.model_name,
            df_categories=num_categories,
            df_num_continuous=len(cont_columns),
            y=y
        )

        print("Training the model...")
        model.fit(
            target_column=y,
            df=processed_df,
            cat_columns=cat_columns,
            cont_columns=cont_columns
        )
        print("Model training completed.")

        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path: str = os.path.join(TRAINED_MODEL_DIR, f"{request.model_name}_{timestamp}")
        model.save(model_path)
        print(f"Model saved at: {model_path}")

        return {"message": f"Model {request.model_name} trained successfully and saved to {model_path}"}

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/test/")
def predict(request: TestPredictionRequest):
    try:
        csv_buffer = io.StringIO(request.csv_data)
        df = pd.read_csv(csv_buffer)

        X = df.copy().drop(columns=[request.target_column])
        y = df[request.target_column].values

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        cleaner = DataCleaner()
        print("Cleaning and processing data...")
        processed_X = cleaner.fit_transform(X)
        print("Data cleaning completed.")

        new_columns = cleaner.numeric_features + cleaner.categorical_features
        processed_df = pd.DataFrame(processed_X, columns=new_columns)

        print(f"Processed data shape: {processed_df.shape}")
        print("Available columns after preprocessing:", processed_df.columns.tolist())

        cont_columns, cat_columns, num_categories = analyze_dataset(processed_df)
        print("Analyzed cleaned data:")
        print("Continuous columns:", cont_columns)
        print("Categorical columns:", cat_columns)
        print("Categories per categorical column:", num_categories)

        model_path = os.path.join(TRAINED_MODEL_DIR, request.model_name)
        print(f"Retrieved model_path: {model_path}")
        model = get_test_model(
            model_path,
            df_categories=num_categories,
            df_num_continuous=len(cont_columns),
            y=y
        )
        print(f"Model loaded from path: {model_path}")

        predictions = model.predict(
            processed_df,
            cat_columns=cat_columns,
            cont_columns=cont_columns
        )
        print("Predictions generated")

        df["predicted_class"] = predictions
        df["actual_class"] = y_encoded
        print(predictions)
        print("Added 'predicted_class' column to DataFrame")

        unique_classes = np.unique(y)

        if len(unique_classes) == 2:
            classification_type = "binary"
            precision = precision_score(y_encoded, predictions, average='binary')
            recall = recall_score(y_encoded, predictions, average='binary')
            f1 = f1_score(y_encoded, predictions, average='binary')
            kappa = cohen_kappa_score(y_encoded, predictions)
            jaccard = jaccard_score(y_encoded, predictions)
            hamming = hamming_loss(y_encoded, predictions)
            accuracy = accuracy_score(y_encoded, predictions)
        else:
            classification_type = "multiclass"
            precision = precision_score(y_encoded, predictions, average='macro')
            recall = recall_score(y_encoded, predictions, average='macro')
            f1 = f1_score(y_encoded, predictions, average='macro')
            accuracy = accuracy_score(y_encoded, predictions)
            kappa = cohen_kappa_score(y_encoded, predictions)
            jaccard = jaccard_score(y_encoded, predictions, average='macro')
            hamming = hamming_loss(y_encoded, predictions)

        accuracy = accuracy_score(y, predictions)

        metrics = ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            classification_type=classification_type,
            kappa=kappa,
            jaccard=jaccard,
            hamming=hamming
        )
        print("Metrics calculated:", metrics.model_dump())

        csv_content = df.to_csv(index=False)
        json_content = metrics.model_dump_json(indent=4)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("predictions.csv", csv_content)
            zip_file.writestr("metrics.json", json_content)

        zip_buffer.seek(0)

        headers = {
            "Content-Disposition": 'attachment; filename="results.zip"'
        }
        return StreamingResponse(zip_buffer, media_type="application/x-zip-compressed", headers=headers)

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


