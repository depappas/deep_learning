import cv2
import numpy as np
import sys
import imageio
import pylab
import time
import readchar

def show_image(title, frame):
    print("frame: " + str(frame))
    fig = pylab.figure()
    fig.suptitle('{}'.format(title), fontsize=20)
    pylab.imshow(frame)
    time.sleep(2)

video = sys.argv[1]

reader = imageio.get_reader(video)

frame = reader.get_next_data()

print("frame: " + str(frame))

r = frame[:,:,2]

g = frame[:,:,1]

b = frame[:,:,0]

show_image('red  :', r)
show_image('green:', g)
show_image('blue :', b)

frame1 = frame
frame1[:,:,1] = 0

show_image('no green  :', frame)

pylab.show()

time.sleep(2)

pylab.close()


