import numpy as np
from keras import backend as K
input = K.placeholder(shape=(2, 4, 5))
val = np.array([[1, 2], [3, 4]])
kvar = K.variable(value=val)
K.ndim(input)
K.ndim(kvar)
input.shape

var.shape

