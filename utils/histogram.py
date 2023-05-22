import numpy as np

class Histogram:
    pass

class Interval:

    def __init__(self, inf: float = 0, sup: float = 20) -> None:
        self.inf = inf
        self.sup = sup

    def count(self, vector: np.ndarray):
        vector = np.array(vector)
        return len(
            vector[
                ((vector > self.inf) & (vector <= self.sup))
            ]
        )
    
    def __repr__(self):
        return "{} -> {}".format(self.inf, self.sup)

class Histogrammer:

    def __init__(self) -> None:
        self.intervals: list[Interval] = []

    def add_interval(self, interval: Interval):
        self.intervals.append(interval)

    def histogrammize(self, vector: list):
        histogram = np.array(list(map(lambda interval: interval.count(vector), self.intervals)))
        return histogram
    
    def __repr__(self):
        return list(map(lambda interval: interval.__repr__(), self.intervals))
    
class Histogram:

    def __init__(self, histogram) -> None:
        self.histogram = histogram

    @property
    def normalized_hist(self):
        return self.histogram / self.histogram.sum()