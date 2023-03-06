from pydantic import BaseModel
import numpy as np
from pathlib import Path

class FeatureSchema(BaseModel):
    feature_vector: np.ndarray
    path: Path
    original_shape: tuple
    layer_name: str

class UserSchema(BaseModel):
    name: str
    dir: Path
    features: list[FeatureSchema]

class MapSchema(BaseModel):
    path: Path
    users: list[UserSchema]