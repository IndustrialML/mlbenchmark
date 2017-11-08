import requests

class Environment(object):

    def __init__(self, name, endpoint, header=None):
        self.name = name
        self.url = endpoint
        self.headers = header if header is not None else {}

    def call(self, data):
        response = requests.post(self.url,
                                 headers=self.headers,
                                 json=self.preprocess_payload(data)
        )

        if response.status_code == 200 : 
            return self.preprocess_response(response)

        else:
            return None

    def preprocess_response(self, response):
        return response.json()

    def preprocess_payload(self, data):
        return data