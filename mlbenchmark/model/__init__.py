import pickle
import numpy as np

def get_image(request_data):
    image = np.array(pickle.loads(request_data))
    return image

