from pydantic import BaseModel
from typing import Optional, List


class DriftConfig(BaseModel):
    target_column: Optional[str] = None
    prediction_column: Optional[str] = None
    ignore_columns: Optional[List[str]] = None


class RetrainConfig(BaseModel):
    target_column: str = "target"
    model_type: str = "random_forest"