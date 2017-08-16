import numpy as np
import cv2
import sys

def show_image(msg, img):
    cv2.imshow(msg, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def get_image(file_name):
    img = cv2.imread(file_name)
    show_image('original image', img)
    return img

def cvt_image_to_grayscale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image('gray scale image', img_gray)
    return img_gray

def threshold_image(img):
    ret, img_thresh = cv2.threshold(img, 127, 255, 0)
    show_image('threshold image', img_thresh)
    return img_thresh 

def get_image_contours(img):
    img_contours, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img_contours, contours, hierarchy

def draw_image_contours(img, contours):
    img_t = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    show_image('contours image', img_t)

def print_rect_coords(img, cnt):
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    im = cv2.drawContours(img,[box],0,(0,0,255),2)
    show_image("Contours img", im)

    M = cv2.moments(cnt)

    if (M['m00'] > 0) :
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
    
        print (leftmost)
        print (rightmost)
        print (topmost)
        print (bottommost)
    
        print(centroid_x)
        print(centroid_y)
    
        print(rect)
        print(box)
    
    
###############################
    
img = get_image(sys.argv[1])
   
img_gray = cvt_image_to_grayscale(img)

img_thresh = threshold_image(img_gray)

img_contours, contours, hierarchy = get_image_contours(img_thresh)

draw_image_contours(img_gray, contours)

img_auto_canny = auto_canny(img_gray)

show_image('auto_canny image', img_auto_canny)

img_contours, contours, hierarchy = get_image_contours(img_auto_canny)

for cnt in contours:
    print_rect_coords(img, cnt)

draw_image_contours(img_gray, contours)





