import time
import os
import sys
import cv2
#import skvideo.io

def getBaseName(file):
  bn = os.path.splitext(file)[0]    
  print("File basename = " + bn)
  return bn
  
def splitVideo(video):
  if not os.path.exists(video):
      print("Exiting: file does not exist: " + video)
      sys.exit()
      
  baseName = getBaseName(video)

  print("baseName: " + baseName)
  
  video = cv2.VideoCapture(video)
  
  #video = skvideo.io.VideoCapture(video)
  if not video:
      print("ERROR: can't read video file!")            
      sys.exit()
      
  
  print(' emitting.....')
  
  i = 0
  # read the file
  while (video.isOpened):
      # read the image in each frame
      success, image = video.read()
      # check if the file has read to the end
      if not success:
          if i == 0:
              print("ERROR: can't read video file!")            
              sys.exit()
          else:
              break;
          
      # convert the image png
      ret, jpeg = cv2.imencode('.png', image)
  
      with open(baseName + '.' + str(i) + '.png', 'wb') as f:
        f.write(data)
      
      if (i % 100):
          print("frame = " + str(i * 100))
      i = i + 1
     
  # clear the capture
  video.release()
  print('done splitting')

print("Args: " + str(sys.argv))

video = sys.argv[1]

print("Video: " + video)

splitVideo(video)
  
