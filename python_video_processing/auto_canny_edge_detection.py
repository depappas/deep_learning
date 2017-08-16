# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
 
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def getContours(img):
    ret,thresh = cv2.threshold(img,127,255,0)
    contours = cv2.findContours(thresh, 1, 2)
    return contours    

def showContour(img, cnt):
    print ("cnt: " + str(cnt.shape))
    print ("cnt len: " + str(len(cnt)))

    img = cv2.drawContours(img, [cnt], 0, (255,0.0), 3)
  # show the images
    cv2.imshow("Original", img)
    cv2.waitKey(0)


    
#     leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
#     rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
#     topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
#     bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
# 
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.cv.BoxPoints(rect)
#     box = np.int0(box)
#     
#     M = cv2.moments(cnt)
#     centroid_x = int(M['m10']/M['m00'])
#     centroid_y = int(M['m01']/M['m00'])
# 
#     print (leftmost)
#     print (rightmost)
#     print (topmost)
#     print (bottommost)
# 
#     print(centroid_x)
#     print(centroid_y)
# 
#     print(rect)
#     print(box)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
    help="path to input dataset of images")
args = vars(ap.parse_args())
 
# loop over the images
for imagePath in glob.glob(args["images"] + "/*.png"):
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
 
    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred)
 
    # show the images
    cv2.imshow("Original", image)
    cv2.imshow("Edges", np.hstack([wide, tight, auto]))
    cv2.waitKey(0)

    contours = getContours(auto)

    for cnt in contours:
      showContour(image, cnt)
      
