import random
from functools import reduce
from pathlib import Path
from .histogram import Histogram
from .user import User
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.user import User
import itertools

class Map:

    def __init__(self, path: Path = None) -> None:
        self.users: list[User] = []
        self.path = path

    def compile(self, 
                sample_rate: float = 1.0,
                ):

        self.sample_rate = sample_rate

        pass

    def add_user(self, user: User):
        self.users.append(user)

    def dict(self):
        return {
            'path': self.path,
            'users': list(map(lambda user: user.dict(), self.users))
        }
    
    def from_dict(self, map_dict):
        map = Map(map_dict['path'])
        for user in map_dict['users']:
            map.add_user(User().from_dict(user))

        return map
    
    def get_user(self, username) -> User:
        return next(filter(lambda user: user.name == username, self.users))
    
    def select_negative_features(self, username, proportion):
        negative_users = list(filter(lambda user: user.name != username, self.users))
        length_user = len(self.get_user(username).features)

        return random.sample(
            reduce(lambda x, y: x + y, list(map(lambda user: user.features, negative_users))), 
            k = proportion * length_user
        )

    def generate_histograms(self, hists):
        return np.array(list(map(lambda hist: Histogram(hist), hists)))
    
    def common_compute_histogram(self, user, distance_vectors, histogrammer, flag='positive'):
        function = user.add_positive_histogram if flag == 'positive' else user.add_negative_histogram
        hists = np.array(list(map(lambda dvec: histogrammer.histogrammize(dvec), distance_vectors)))

        histograms = self.generate_histograms(hists)
        for i in range(histograms.shape[0]):
            function(histograms[i])

        return histograms
    
    def compute_user_positive_histograms(self, user: User, histogrammer):

        pos = user.positive_distance_vectors()

        return self.common_compute_histogram(user, pos, histogrammer)
        
    
    def compute_user_negative_histograms(self, user: User, histogrammer, proportion = 3):

        negs = user.negative_distance_vectors(self, proportion)

        return self.common_compute_histogram(user, negs, histogrammer, 'negative')
    
    
    def compute_dataframe(self, histogrammer) -> None:

        # both return an object like:
        # p: list[list[list]] = [[[Hist, Hist, Hist], [Hist, Hist, Hist]], [[Hist, Hist, Hist], [Hist, Hist, Hist]]]
        positives = list(map(lambda user: user.positive_histograms, self.users))
        negatives = list(map(lambda user: user.negative_histograms, self.users))

        # both now are like p: list[list] = [Hist, Hist, Hist], [Hist, Hist, Hist]
        positives: list[list] = list(itertools.chain.from_iterable(positives))
        negatives: list[list] = list(itertools.chain.from_iterable(negatives))

        # samples list is like p: list[list] = [Hist, Hist, Hist]
        samples: list[list] = list(itertools.chain.from_iterable([positives, negatives]))
        # samples list now is like p: list[list] = [[...], [...], [...]]
        samples = list(map(lambda sample: sample.histogram, samples))

        # create labels list
        target: list[int] = list(
            itertools.chain.from_iterable(
                [np.ones(len(positives)), 
                np.zeros(len(negatives))]
            )
        )

        # dataframe is an object like:
        #
        # 0->6, 6->7, 7->8, ..., 18->50  target
        #
        #   1     4     3       0   0      1.0
        #   2     3     3       0   0      1.0
        #   ...         ...         ...    ...
        #   0     0     0   ... 5   1      0.0

        self.df = pd.DataFrame(samples, columns=histogrammer.__repr__())

        self.standad_scaler = StandardScaler().fit(self.df)
        self.min_max_scaler = MinMaxScaler().fit(self.df)
        self.df["target"] = target

        return      
    
    def compute_histograms(self, histogrammer, proportion = 3):
        for user in self.users:
            self.compute_user_positive_histograms(user, histogrammer)
            self.compute_user_negative_histograms(user, histogrammer, proportion)

    def reset_histograms(self):
        for user in self.users:
            user.reset()


class TestMap(Map):

    def __init__(self, path: Path = None) -> None:
        super().__init__(path)

    def negative_users_per_user(self):
        """
        users and its negatives
        return:
        [
            (user1, [user2, user3, ...]),
            (user2, [user1, user3, ...]),
            ...
            (userN, [user1, user2, ...])
        ]
        
        """
        return list(
            map(
                lambda tup: (
                                self.users[tup[0]],
                                np.concatenate((self.users[:tup[0]], self.users[tup[0]+1:]))
                            ), 
                enumerate(self.users)
            )
        )
    
    def compute_dataframe(self, histogrammer):
        users_and_their_negatives: list[tuple[User, list[User]]] = self.negative_users_per_user()
        negative_features = []
        names = ['']

        positive_features = []
        for user in self.users:
            p = user.positive_distance_vectors()
            positive_features.append(p)
            names += [user.name for _ in range(len(p))]

        for (user, negatives) in users_and_their_negatives:
            for anchor_feature in user.features:
                for negative_user in negatives:
                    n = anchor_feature @ negative_user.features
                    negative_features.append(n)

                    names += [user.name]

        positive_features = list(itertools.chain.from_iterable(positive_features))
        # samples list is like p: list[list] = [Hist, Hist, Hist]
        samples: list[list] = list(itertools.chain.from_iterable([positive_features, negative_features]))
        # create labels list
        target: list[int] = list(
            itertools.chain.from_iterable(
                [np.ones(len(positive_features)), 
                np.zeros(len(negative_features))]
            )
        )

        samples = list(map(lambda feature: histogrammer.histogrammize(feature), samples))
        self.df = pd.DataFrame(samples, columns=histogrammer.__repr__())
        self.df["user"] = names[1:]
        self.df["target"] = target