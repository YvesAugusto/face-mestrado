import random
from pathlib import Path

import numpy as np

from .feature import Feature
from .histogram import Histogram


class User:

    def __init__(self, name: str = None, dir: Path = None) -> None:
        self.name = name
        self.features: list[Feature] = []
        self.positive_histograms: list[Histogram] = []
        self.negative_histograms: list[Histogram] = []
        self.dir = dir

    def add_feature(self, feature: Feature):
        self.features.append(feature)

    def add_positive_histogram(self, histogram: Histogram):
        self.positive_histograms.append(histogram)

    def add_negative_histogram(self, histogram: Histogram):
        self.negative_histograms.append(histogram)

    def positive_distance_vectors(self):
        features = np.array(self.features)
        M = []
        for idf, feature in enumerate(features):
            other_features = np.concatenate((features[0:idf], features[idf + 1:]))
            M.append(feature @ other_features)

        return np.array(M)
    
    def negative_distance_vectors(self, data_map, proportion):
        # data_map: Map type

        negative_samples = data_map.select_negative_features(self.name, proportion)
        
        features = np.array(self.features)
        M = []
        for idf, feature in enumerate(negative_samples):
            other_features = random.sample(list(features), k=features.shape[0] - 1)
            M.append(feature @ other_features)

        return np.array(M)
    
    def dict(self):
        return {
            'name': self.name,
            'features': np.array(list(map(lambda feature: feature.dict(), self.features))),
            'dir': self.dir
        }

    def from_dict(self, user_dict):
        user = User(
            name=user_dict['name'],
            dir=user_dict['dir']
        )
        for feature in user_dict['features']:
            user.add_feature(Feature().from_dict(feature))

        return user