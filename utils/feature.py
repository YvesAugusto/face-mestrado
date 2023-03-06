from pathlib import Path
import numpy as np
from numpy.linalg import norm as L2

class Feature:

    def __init__(
            self, feature_vector: np.ndarray, 
            path: Path,
            original_shape: tuple,
            layer_name: str = 'last_bn_norm'
    ) -> None:

        self.feature_vector = feature_vector
        self.path = path
        self.original_shape = original_shape
        self.layer_name = layer_name

    def __sub__(self, alium) -> float:
        
        return L2(self.feature_vector - alium.feature_vector)
    
    def __matmul__(self, alia_features):

        distances_vector = list(map(self.__sub__, alia_features))
        return distances_vector