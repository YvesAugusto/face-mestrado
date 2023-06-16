from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def find_best_parameters_random_forest(X, Y):
    grid_params = {
        'n_estimators': [20, 40, 60, 80, 100, 120, 140],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 3, 5, 7, 9, 11],
        'max_depth': [5, 10, 20, 30, 40, None]
    }

    cv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=grid_params, cv=4)
    cv.fit(X, Y)

    print()