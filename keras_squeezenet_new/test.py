import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.utils import plot_model


model = SqueezeNet()

plot_model(model, to_file='model.png')

print(model.summary())
