import argparse
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from utils.data import load_pickle
from utils.histogram import Histogrammer, Interval
from utils.map import Map, TestMap

BASE_PATH = Path(pathlib.Path(__file__).parent.resolve()) / "features"
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", required=True,
                    help="nome do arquivo pickle no qual a base de dados foi codificada")
args = parser.parse_args()

histogrammer = Histogrammer()
intervals = [
    Interval(0, 6), Interval(6, 7), Interval(7, 8), 
    Interval(8, 9), Interval(9, 10), Interval(10, 11), 
    Interval(11, 12), Interval(12, 13), Interval(13, 14), 
    Interval(14, 15), Interval(15, 16), Interval(16, 17),
    Interval(17, 18), Interval(18, 50)
]
for interval in intervals:
    histogrammer.add_interval(interval)

if __name__ == '__main__':

    filepath: Path = BASE_PATH / args.file
    train_map: Map = load_pickle(filepath)

    train_map.compute_histograms(histogrammer, 2)
    train_map.compute_dataframe(histogrammer)
    
    X = train_map.min_max_scaler.transform(train_map.df.drop('target', axis=1))
    Y = train_map.df['target']

    clf = RandomForestClassifier()
    clf.fit(X, Y)

    test_filepath = Path('/' + '/'.join(filepath.parts[1:-1])) / "test-0.pickle"

    test_map: TestMap = load_pickle(test_filepath)

    test_map.compute_histograms(histogrammer, 2)
    test_map.compute_dataframe(histogrammer)

    X_test = train_map.min_max_scaler.transform(test_map.df.drop(['target', 'user'], axis=1))
    Y_test = test_map.df['target']
    users = test_map.df['user']

    y_pred = clf.predict(X_test)
    df = test_map.df.copy()
    df['y_pred'] = y_pred

    groups = df.groupby(['user', 'target']).apply(lambda x: (x['target'] != x['y_pred']).mean())
    result = {"user": [], "fp": [], "fn": []}
    for user in np.unique(list(map(lambda x: x[0], list(groups.index)))):
        result["user"].append(user)
        result["fp"].append(groups[user][0.0])
        result["fn"].append(groups[user][1.0])

    result_df = pd.DataFrame(result)
    print()