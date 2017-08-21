import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

model = SqueezeNet()

#img = image.load_img('images/cat.jpeg', target_size=(227, 227))
#img = image.load_img('images/basketball.jpeg', target_size=(227, 227))
#img = image.load_img('images/teapot.jpg', target_size=(227, 227))
# = image.load_img('images/bb_defense_rim.jpg', target_size=(227, 227))
# img = image.load_img('images/mixer1.jpg', target_size=(227, 227))
# img = image.load_img('images/sewing_machine.jpeg', target_size=(227, 227))
# img = image.load_img('images/soccer_cross_right.jpg', target_size=(227, 227))
img = image.load_img('images/s1.jpg', target_size=(227, 227))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))

