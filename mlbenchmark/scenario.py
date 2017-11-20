import multiprocessing
import time
from concurrent.futures.process import ProcessPoolExecutor

import numpy as np
from sklearn import metrics

from collections import namedtuple

AccuracyResult = namedtuple("AccuracyResult", ["accuracy", "jaccard"])
TimingResult = namedtuple("TimingResult", ["count", "mean", "median", "min", "max", "std", "geomean"])

class BenchmarkScenario(object):

    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.name = type(self).__name__

    def benchmark(self, env):
        pass


class AccuracyBenchmark(BenchmarkScenario):

    def benchmark(self, env):
        y_true = self.data_provider.y_true

        preds = []
        for x_test, y_test in self.data_provider:
            pred = env.call(x_test)

            if pred is None:
                pred = -1

            preds.append(pred)

        return AccuracyResult(metrics.accuracy_score(y_true, preds),
                              metrics.jaccard_similarity_score(y_true, preds),
                              )


def timing_wrapper(env, data):
    t0 = time.time()
    env.call(data)
    t1 = time.time()
    return (t1 - t0)


class TimingBenchmark(BenchmarkScenario):

    def process_timings(self, timings):
        timings = np.array([t for t in timings if t is not None])
        return TimingResult(len(timings),
                            np.mean(timings),
                            np.median(timings),
                            np.min(timings),
                            np.max(timings),
                            np.std(timings),
                            np.exp(np.log(timings).sum() / len(timings))
                            )


class SequentialLoadBenchmark(TimingBenchmark):

    def benchmark(self, env):
        timings = [timing_wrapper(env, x_test) for x_test, y_test in self.data_provider]
        return self.process_timings(timings)


class ConcurrentLoadBenchmark(TimingBenchmark):

    def __init__(self, data_provider, ncores=None):
        super(ConcurrentLoadBenchmark, self).__init__(data_provider)
        self.ncores = ncores if ncores is not None else multiprocessing.cpu_count()

    def benchmark(self, env):

        with ProcessPoolExecutor(self.ncores) as executor:
        # with ThreadPoolExecutor(self.ncores) as executor:
            res = [executor.submit(timing_wrapper, env, x_test) for x_test, y_test in self.data_provider]

        return self.process_timings([r.result() for r in res])