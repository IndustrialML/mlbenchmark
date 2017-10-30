import json
import pytest
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

from mlbenchmark import scenario
from mlbenchmark.data import DataProvider
from mlbenchmark.environment import Environment
from .support import ResultCollector

ENVIRONMENTS = [
    ("local-python-baseline",   "http://localhost:5000/baseline/predict_digits/"),
    # ("local-python-svm",        "http://localhost:5001/svm/predict_digits/"),
    ("local-python-forest",     "http://localhost:5002/forest/predict_digits/"),
    ]

SCENARIOS =[
    scenario.AccuracyBenchmark,
    scenario.SequentialLoadBenchmark,
    scenario.ConcurrentLoadBenchmark
    ]


class MNistEnvironment(Environment):

    def preprocess_payload(self, data):
        return json.dumps(data.flatten().tolist())


@pytest.fixture
def mnist_digits():
    mnist = fetch_mldata('MNIST original')
    data = mnist.data[:2000]
    target = mnist.target[:2000]
    _, x_test, _, y_test = train_test_split(data, target, test_size = 0.33, random_state = 42)
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