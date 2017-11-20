from flask import Flask, request
import numpy as np
import pickle

from mlbenchmark.model import get_image

app = Flask(__name__)


@app.route("/baseline/predict_digits/", methods=["POST"])
def predict_digit():
    request_data = request.get_json()
    image = get_image(request_data)

    return "%s"%np.random.randint(0, 9)


if __name__ == '__main__':
    app.run()