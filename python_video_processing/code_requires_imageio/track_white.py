# https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv

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


print(str(reader))
#cap = cv2.VideoCapture(0)

while(1):

    frame = reader.get_next_data()

#    _, frame = cap.read()

    print("frame: " + str(frame))
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([0,0,255], dtype=np.uint8)

    lower_black = np.array([0,0,0], dtype = "uint16")
    upper_black = np.array([70,70,70], dtype = "uint16")

    
    # Threshold the HSV image to get only white colors
#    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    show_image('frame',frame)
    show_image('mask',mask)
    show_image('res',res)
    
    print("Reading a char:")
    pylab.show()
    print(repr(readchar.readchar()))
    pylab.close()

cv2.destroyAllWindows()
