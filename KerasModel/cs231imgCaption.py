import time, os, json
import numpy as np
import matplotlib.pyplot as plt
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.rnn_layers import *
from cs231n.captioning_solver import CaptioningSolver
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url

# SETTING DEFAULT SIZE OF PLOTS
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data = load_coco_data(pca_features=True)
for k, v in data.iteritems():
    if type(v) == np.ndarray:
        print k, type(v), v.shape, v.dtype
    else:
        print k, type(v), len(v)