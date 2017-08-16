
import numpy as np

import imageio
import sys

video = sys.argv[1]

reader = imageio.get_reader(video)

im = reader.get_next_data()

print(im)
print(type(im))

n = np.frombuffer(im, dtype="uint8")

print(im)
print(im.shape)

