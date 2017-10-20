import pickle
import pytest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from mlbenchmark import scenario
from mlbenchmark.data import DataProvider
from mlbenchmark.environment import Environment
from .support import ResultCollector

ENVIRONMENTS = [
    ("local-python-baseline",   "http://localhost:5000/baseline/predict_digits/"),
    ("local-python-svm",        "http://localhost:5001/svm/predict_digits/"),
    ("local-python-forest",     "http://localhost:5002/forest/predict_digits/"),
    ]

SCENARIOS =[
    scenario.AccuracyBenchmark,
    scenario.SequentialLoadBenchmark,
    scenario.ConcurrentLoadBenchmark
    ]


class MNistEnvironment(Environment):

    def preprocess_payload(self, data):
        return pickle.dumps(data.tolist(), protocol=3)


@pytest.fixture
def mnist_digits():
    digits = load_digits()
    _, x_test, _, y_test = train_test_split(digits.images, digits.target, test_size = 0.33, random_state = 42)
    return DataProvider(x_test, y_test)


@pytest.fixture(params=SCENARIOS)
def scenario(request, mnist_digits):
    return request.param(mnist_digits)


@pytest.fixture(scope="module")
def result_collector():
    writer = ResultCollector()

    yield writer

    writer.finalize()


@pytest.mark.parametrize("env", ENVIRONMENTS)
def test_mnist_digists(scenario, env, result_collector):
    env = MNistEnvironment(*env)
    result = scenario.benchmark(env)

    result_collector.collect(scenario, env, result)