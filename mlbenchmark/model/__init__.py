import json
import numpy as np


def get_image(request_data):
    image = np.array(request_data)
    size = int(np.sqrt(image.shape[0]))
    return image.reshape((size, size))

