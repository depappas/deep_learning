# https://stackoverflow.com/questions/43139764/perform-object-recognition-based-on-colour

import numpy as np
import cv2
cap = cv2.VideoCapture('/home/depappas/Dropbox/soccer_images/video_test/t.mp4')
fgbg =  cv2.createBackgroundSubtractorMOG2()

print("fgbg: " + str(fgbg))

j=0
count = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print("count: " + str(count))

while j < count:
    ret, frame = cap.read()
    cmask = fgbg.apply(frame)
    fgmask = cmask.copy()
    floodfill =cmask.copy()

    (cnts, _) = cv2.findContours(cmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    if(len(cnts)!=0):
        h, w = floodfill.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(floodfill, mask, (0,0), 255)
        floodfill_inv = cv2.bitwise_not(floodfill)
        fgmask=fgmask|floodfill_inv

    # screenCnt = None
    print("K="+str(j))
    j+=1
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(fgmask,(x,y),(x+w,y+h),255,4)
    if(len(cnts)!=0):
        cv2.imshow('frame',fgmask)

    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break
cap.release()
