from pydantic import BaseModel


class TrainPredictionRequest(BaseModel):
    model_name: str
    target_column: str
    csv_data: str


class TestPredictionRequest(BaseModel):
    model_name: str
    csv_data: str
    target_column: str

