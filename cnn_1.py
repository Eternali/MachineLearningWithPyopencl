import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot


def cnn_model (features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
    


