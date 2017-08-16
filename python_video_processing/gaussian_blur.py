import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/depappas/Dropbox/soccer_images/test1/t.png',1) # 0 arg means convert to grayscale

kernel_size = 5

img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

img_canny = cv2.Canny(img_blur, low_threshold, high_threshold)

plt.subplot(121),plt.imshow(img_blur,cmap = 'gray')
plt.title('Image blur'), plt.xticks([]), plt.yticks([])

plt.subplot(121),plt.imshow(img_canny,cmap = 'gray')
plt.title('Image canny'), plt.xticks([]), plt.yticks([])



#edges = cv2.Canny(img_gray_scale,100,200)
#
#ret,thresh = cv2.threshold(edges,127,255,0)
##contours = cv2.findContours(thresh, 1, 2)
#contours = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
#
#print("contours: " + str(len(contours)))
#
#for cnt in contours:
##  cnt = contours[0]
#  x,y,w,h = cv2.boundingRect(cnt)
#  #print ("x, y, x+w, y+h: " + str(x), str(y), str(x+w), str(y+h)) 
#  img_bb = cv2.rectangle(img_gray_scale,(x,y),(x+w,y+h),(0,255,0),2)
#
#plt.subplot(121),plt.imshow(img_gray_scale,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(img_bb,cmap = 'gray')
#plt.title('Image BB'), plt.xticks([]), plt.yticks([])
#
#plt.show()
#
#
#cv2.drawContours(img_gray_scale, contours, -1, (0,255,0), 3)
