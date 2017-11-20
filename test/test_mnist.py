import json
import numpy as np
import pytest
import requests
from sklearn.datasets import fetch_mldata

from mlbenchmark import scenario
from mlbenchmark.data import DataProvider
from mlbenchmark.environment import Environment
from .support import ResultCollector


class MNistEnvironment(Environment):

    def preprocess_payload(self, data):
        return data.flatten().tolist()

class OpencpuMNistEnv(Environment):
    def __init__(self, name, endpoint):
        super().__init__(name, endpoint, {"Content-Type": "application/json"})

    def preprocess_payload(self, data):
        return { "image": data.flatten().tolist()}


class MSMLServerMNistEnv(Environment):

    def __init__(self, name, endpoint, login_endpoint, username, password):
        header = self.create_auth_header(login_endpoint, username, password)
        super(MSMLServerMNistEnv, self).__init__(name, endpoint, header=header)

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
        label = payload["outputParameters"]["label"]
        return int(label)


class MSMLServerRealtimeMNistEnv(MSMLServerMNistEnv):

    def preprocess_payload(self, data):
        encoded_image = {}
		
        for i, entry in enumerate(data.flatten().tolist()):
            encoded_image["V%s"%(i+1)] = [entry]
	
        return {"inputData": encoded_image}

    def preprocess_response(self, response):
        payload = response.json()
        label = payload["outputParameters"]["outputData"]["Y_Pred"][0]
        return int(label)




ENVIRONMENTS = [
    # MNistEnvironment("local-python-baseline", endpoint="http://localhost:5000/baseline/predict_digits/"),
    # MNistEnvironment("local-python-svm", endpoint="http://localhost:5001/svm/predict_digits/forest_50"),
    # MNistEnvironment("local-python-forest", endpoint="http://localhost:5002/forest/predict_digits/"),
	# MNistEnvironment("python-vm-baseline", endpoint="http://lin-mlserver.westeurope.cloudapp.azure.com:8003/predict"),
    #MNistEnvironment("python-vm-forest_50", endpoint="http://lin-mlserver.westeurope.cloudapp.azure.com:8004/predict"),
    #MNistEnvironment("python-vm-forest_500", endpoint="http://lin-mlserver.westeurope.cloudapp.azure.com:8005/predict"),
    # MNistEnvironment("R-plumber-local-empty", endpoint="http://localhost:8080/predictemptypkg"),
	# MNistEnvironment("R-plumber-local-small", endpoint="http://localhost:8080/predictsmallpkg"),
	# MNistEnvironment("R-plumber-local-large", endpoint="http://localhost:8080/predictlargepkg"),
	# MNistEnvironment("R-plumber-vm-empty", endpoint="http://lin-mlserver.westeurope.cloudapp.azure.com:8080/predictemptypkg"),
	# MNistEnvironment("R-plumber-vm-small", endpoint="http://lin-mlserver.westeurope.cloudapp.azure.com:8080/predictsmallpkg"),
	# MNistEnvironment("R-plumber-vm-large", endpoint="http://lin-mlserver.westeurope.cloudapp.azure.com:8080/predictlargepkg"),
	# OpencpuMNistEnv("R-opencpu-local-empty", endpoint="http://localhost:80/ocpu/library/digiterEmpty/R/predict_digit_empty/json"),
	# OpencpuMNistEnv("R-opencpu-local-small", endpoint="http://localhost:80/ocpu/library/digiterSmall/R/predict_digit_small/json"),
	# OpencpuMNistEnv("R-opencpu-local-large", endpoint="http://localhost:80/ocpu/library/digiterLarge/R/predict_digit_large/json"),
	# OpencpuMNistEnv("R-opencpu-vm-empty", endpoint="http://lin-mlserver.westeurope.cloudapp.azure.com:80/ocpu/library/digiterEmpty/R/predict_digit_empty/json/"),
	# OpencpuMNistEnv("R-opencpu-vm-small", endpoint="http://lin-mlserver.westeurope.cloudapp.azure.com:80/ocpu/library/digiterSmall/R/predict_digit_small/json/"),
	# OpencpuMNistEnv("R-opencpu-vm-large", endpoint="http://lin-mlserver.westeurope.cloudapp.azure.com:80/ocpu/library/digiterLarge/R/predict_digit_large/json/"),
    # MSMLServerMNistEnv("MS ML Server small", 
    #                  "http://lin-mlserver.westeurope.cloudapp.azure.com:12800/api/modelSmall_transp/v1.0.0",
    #                  "http://lin-mlserver.westeurope.cloudapp.azure.com:12800/login", 
	#                  username="admin", password="PwF/uOnBo1"),
	# MSMLServerMNistEnv("MS ML Server empty",
    #                 "http://lin-mlserver.westeurope.cloudapp.azure.com:12800/api/modelEmpty/v1.0.0",
    #                 "http://lin-mlserver.westeurope.cloudapp.azure.com:12800/login",
    #                  username="admin", password="PwF/uOnBo1"),
	MSMLServerRealtimeMNistEnv("MS ML Server Realtime small",
	                 "http://lin-mlserver.westeurope.cloudapp.azure.com:12800/api/rxDModelsmall/v1.0.0",
                     "http://lin-mlserver.westeurope.cloudapp.azure.com:12800/login",
                      username="admin", password="PwF/uOnBo1"),
    MSMLServerRealtimeMNistEnv("MS ML Server Realtime large",
	                 "http://lin-mlserver.westeurope.cloudapp.azure.com:12800/api/rxDModellarge/v1.0.0",
                     "http://lin-mlserver.westeurope.cloudapp.azure.com:12800/login",
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

    size = 10
    idx = np.random.choice(mnist.data.shape[0], size)
    x_test = mnist.data[idx]
    y_test = mnist.target[idx]

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