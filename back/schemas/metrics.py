from pydantic import BaseModel


class ClassificationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    classification_type: str
    kappa: float
    jaccard: float
    hamming: float