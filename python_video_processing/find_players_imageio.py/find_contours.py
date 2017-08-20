from matplotlib import pyplot as plt
import imutils

import numpy as np
import cv2
image = cv2.imread('/home/depappas/Dropbox/soccer_images/test1/t.png')
resized = imutils.resize(image, width=300)
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

# convert the resized image to grayscale, blur it slightly,
# and threshold it
#gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
 
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

plt.subplot(121)
plt.imshow(image,cmap = 'gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(im2,cmap = 'gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])

plt.show()
