import argparse
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from utils.optimizer import RFOptimizer
from utils.data import load_pickle
from utils.histogram import Histogrammer, Interval
from utils.map import Map, TestMap
from utils.tools import find_best_parameters_random_forest

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

def read_files(args):
    filepath: Path = BASE_PATH / args.file
    number = filepath.parts[-1].split(".")[0].split("-")[-1]

    train_map: Map = load_pickle(filepath)

    test_filepath = Path('/' + '/'.join(filepath.parts[1:-1])) / "test-{}.pickle".format(number)
    test_map: TestMap = load_pickle(test_filepath)

    return (train_map, filepath), (test_map, test_filepath)

def make_train_dataframe(train_map: Map):

    train_map.reset_histograms()
    train_map.compute_histograms(histogrammer, 2)
    train_map.compute_dataframe(histogrammer)

def preprocess(map_scaler, map, drop=['target']):
    return map_scaler.min_max_scaler.transform(map.df.drop(drop, axis=1)), map.df['target']

def test(test_map, X_test):

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

    return result_df

if __name__ == '__main__':

    optimizer = RFOptimizer()
    (train_map, filepath), (test_map, test_filepath) = read_files(args)

    make_train_dataframe(train_map)
    test_map.compute_dataframe(histogrammer)
    X, Y = preprocess(train_map, train_map, ['target'])
    # find_best_parameters_random_forest(X, Y)
    optimizer.find_best_params(X, Y)

    fn_ = []
    fp_ =[]
    test_map.compute_dataframe(histogrammer)
    for _ in range(10):
        make_train_dataframe(train_map)
        X, Y = preprocess(train_map, train_map, drop=['target'])
        X_test, Y_test = preprocess(train_map, test_map, drop=['target', 'user'])

        clf = RandomForestClassifier(
            # max_depth=30, min_samples_leaf=5, min_samples_split=6,
            # n_estimators=20
        )
        clf.fit(X, Y)

        results = test(test_map, X_test)
        fn, fp = results["fn"].mean() * 100, results["fp"].mean() * 100
        fn_.append(fn)
        fp_.append(fp)

        print("False Positive: {}, False Negative: {}".format(fp, fn))

    print("Mean False Positive: {}, Mean False Negative: {}".format(np.mean(fp_), np.mean(fn_)))
        