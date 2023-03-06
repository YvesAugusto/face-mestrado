from pathlib import Path
import numpy as np
from .feature import Feature

class User:

    def __init__(self, name: str, dir: Path) -> None:
        self.name = name
        self.features: list[Feature] = []
        self.dir = dir

    def add_feature(self, feature: Feature):
        self.features.append(feature)