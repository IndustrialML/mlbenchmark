from flask import Flask, request
from sklearn import datasets, svm, model_selection

from model import get_image

model = None

app = Flask(__name__)


@app.route("/svm/predict_digits/", methods=["POST"])
def predict_digit():
    request_data = request.get_data()
    image = get_image(request_data)

    pred = model.predict(image.reshape(1, -1))
    return "%s"%pred


def prepare_model():

    digits = datasets.load_digits()
    classifier = svm.SVC(gamma=0.001)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(digits.images, digits.target,
                                                                        test_size=0.33, random_state=42)
    train_reshape = x_train.reshape(((len(x_train)), -1))
    classifier.fit(train_reshape, y_train)

    global model
    model = classifier


if __name__ == '__main__':
    prepare_model()
    app.run(port=5001)