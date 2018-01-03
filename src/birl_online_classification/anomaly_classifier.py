import numpy as np
from keras.models import load_model
import pandas as pd
import os

# if you want to run this code on your computer, pls change this PATH

dir_path = os.path.dirname(os.path.realpath(__file__))

def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]

def run(np_matrix):
    model = load_model(os.path.join(dir_path, '..', '..', 'models', 'model_Anomaly_classification.h5'))
    '''
    @ np_matrix: 2D numpy array, shape must be (40, 13).
    ''' 
    assert np_matrix.shape == (40, 13)
    np_array = np.asarray(np_matrix).reshape(1,40,13)

    pred_destribution = model.predict(np_array) 
    pred_label = one_hot_decode(pred_destribution)

    return pred_label[0], pred_destribution[0].max()
