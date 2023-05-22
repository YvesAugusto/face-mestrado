from pathlib import Path
import numpy as np
from numpy.linalg import norm as L2

class Feature:

    def __init__(
        self, feature_vector: np.ndarray = None, 
        path: Path = None,
        original_shape: tuple = None,
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
    
    def dict(self):
        return {
            'feature_vector': self.feature_vector,
            'path': self.path,
            'original_shape': self.original_shape,
            'layer_name': self.layer_name
        }
    
    def from_dict(self, feature_dict):
        return Feature(
            feature_vector=feature_dict['feature_vector'],
            path=feature_dict['path'],
            original_shape=feature_dict['original_shape'],
            layer_name=feature_dict['layer_name']
        )