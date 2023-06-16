import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics._scorer import make_scorer
from sklearn.model_selection import GridSearchCV

from constants import RF_GRID_PATH


class CVResult:

    def __init__(self, cv) -> None:
        self.cv = cv

class Optimizer:

    def __init__(self) -> None:
        pass

    def read_grid_params(self, path):
        with open(path, "rb") as file:
            grid = json.load(file)
        return grid
    
    def score_function(self, *args, **kwargs):
        y_true = args[0].values
        y_pred = args[1]

        negative_true = y_true[y_true == 0]
        preds = y_pred[y_true == 0]
        fp_rate = np.mean(preds != negative_true)

        positive_true = y_true[y_true == 1]
        preds = y_pred[y_true == 1]
        fn_rate = np.mean(preds != positive_true)
        print(args, kwargs, flush=True)
        exit(1)

    def find_best_params(self, X, Y, estimator_class, grid_params):
        cv = GridSearchCV(estimator=estimator_class(), param_grid=grid_params, cv=4, scoring=make_scorer(self.score_function))
        cv.fit(X, Y)

        return cv

class RFOptimizer(Optimizer):
    def __init__(self) -> None:
        super().__init__()

    @property
    def grid_params(self):
        return super().read_grid_params(RF_GRID_PATH)
    
    def find_best_params(self, X, Y):
        cv = super().find_best_params(X, Y, RandomForestClassifier, self.grid_params)
        