import json
import pytest
import requests
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

from mlbenchmark import scenario
from mlbenchmark.data import DataProvider
from mlbenchmark.environment import Environment
from .support import ResultCollector


class MNistEnvironment(Environment):

    def preprocess_payload(self, data):
        return data.flatten().tolist()


class MSRServerMNistEnv(Environment):

    def __init__(self, name, endpoint, login_endpoint, username, password):
        header = self.create_auth_header(login_endpoint, username, password)
        super(MSRServerMNistEnv, self).__init__(name, endpoint, header=header)

    def create_auth_header(self, login_endpoint, username, password):
        response = requests.post(login_endpoint,
                                headers={"Content-Type": "application/json"},
                                json={"username": username,
                                      "password": password}
                                ).json()

        token = response["access_token"]

        return {"Authorization": "Bearer %s"%token,
                "Content-Type": "application/json"}

    def preprocess_payload(self, data):
        return {"dataframe_transp": {
            "image": data.flatten().tolist()
        }}

    def preprocess_response(self, response):
        payload = response.json()
        return payload["outputParameters"]["label"]


ENVIRONMENTS = [
    # MNistEnvironment("local-python-baseline", endpoint="http://localhost:5000/baseline/predict_digits/"),
    # MNistEnvironment("local-python-svm",     endpoint="http://localhost:5001/svm/predict_digits/"),
    # MNistEnvironment("local-python-forest",   endpoint="http://localhost:5002/forest/predict_digits/"),
    MSRServerMNistEnv("MS R Server small",
                      "http://lin-op-vm.westeurope.cloudapp.azure.com:12800/api/modelSmall_transp/v1.0.0",
                      "http://lin-op-vm.westeurope.cloudapp.azure.com:12800/login",
                      username="admin", password="PwF/uOnBo1"),
    ]

SCENARIOS =[
    scenario.AccuracyBenchmark,
    scenario.SequentialLoadBenchmark,
    scenario.ConcurrentLoadBenchmark
    ]



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
    result = scenario.benchmark(env)

    result_collector.collect(scenario, env, result)