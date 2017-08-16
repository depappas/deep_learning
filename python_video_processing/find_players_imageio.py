import sys
import numpy as np
import scipy as sp
import scipy.ndimage
import pylab

import imageio

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
    
def animate(input_filename):
    """Detects objects  animates the position and raw data of each frame
    in the input file"""
    # With matplotlib, it's much, much faster to just update the properties
    # of a display object than it is to create a new one, so we'll just update
    # the data and position of the same objects throughout this animation...

    reader = imageio.get_reader(input_filename,  'ffmpeg')
#    reader = read_file(input_filename)
    # Since we're making an animation with matplotlib, we need 
    # ion() instead of show()...
    #plt.ion()
    #fig = plt.figure()
    ##ax = fig.add_subplot(111)
    #fig.suptitle(input_filename)

    # Make an image based on the first frame that we'll update later
    # (The first frame is never actually displayed)
#    im = ax.imshow(reader.get_data(7000))

    # Make 4 rectangles that we can later move to the position of each object
    rects = [Rectangle((0,0), 1,1, fc='none', ec='red') for i in range(20)]
    #[ax.add_patch(rect) for rect in rects]

    #title = ax.set_title('Time 0.0 ms')
    
    #nums = [7000, 7005]

    show_frames(reader, rects)
    
def view_frame(reader, rects):
    for num in range(7060,7066, 2): 
        image = reader.get_data(num)
        fig = pylab.figure()
        fig.suptitle('image #{}'.format(num), fontsize=20)
        pylab.imshow(image)
        time.sleep(2)
    pylab.show()
    
    
def show_frames(reader, rects):
    """ Process and display each frame
    """
    i = 0
    for i in range(7000, 7017, 8):
        frame = reader.get_data(i)
#        frame = reader.get_data(i)
        #show_image(frame, i)
        show_frame(frame, rects)
        
def show_frame(frame, rects):
    #print(frame)

    print(type(frame))
    print("shape: " + str(frame.shape))
 
    n = np.frombuffer(frame, dtype="uint8")

    object_slices = find_objects(frame)
    print("object_slices: " + str(object_slices))

    # Hide any rectangles that might be visible
    [rect.set_visible(False) for rect in rects]

    # Set the position and size of a rectangle for each object and display it
    for slice, rect in zip(object_slices, rects):
        print(rect)
        dy, dx = slice
        rect.set_xy((dx.start, dy.start))
        rect.set_width(dx.stop - dx.start + 1)
        rect.set_height(dy.stop - dy.start + 1)
        rect.set_visible(True)

    # Update the image data and title of the plot
    # title.set_text('Time %0.2f ms' % time)
    title.set_text('Frame #: %0.2d' % 1)
    im.set_data(frame)
    im.set_clim([frame.min(), frame.max()])
    fig.canvas.draw()

def show_image(frame, i):
    print ("show_image: " + str(i))
    fig = pylab.figure()
    fig.suptitle('image #{}'.format(i), fontsize=20)
    pylab.imshow(frame)
    pylab.show()
    #time.sleep(2)
    pylab.close()

def print_array(a):
    print("Type : " + str(type(a)))
    print("Shape: " + str(a.shape))
    print("Vals : " + str(a[0:1]))
    
def find_objects(data, smooth_radius=5, threshold=0.0001):
    """Detects and isolates contiguous regions in the input array"""

    show_image(data, 0)
    print_array(data)

    # Blur the input data a bit so the objects have a continous footprint
    data = sp.ndimage.uniform_filter(data, smooth_radius)
    print_array(data)
    show_image(data, 1)

    # Threshold the blurred data (this needs to be a bit > 0 due to the blur)
    thresh = data > threshold

    # Fill any interior holes in the objects to get cleaner regions...
    filled = sp.ndimage.morphology.binary_fill_holes(thresh)
    np.set_printoptions(threshold='nan')
    print_array(filled)
    #show_image(filled, 2)

    # Label each contiguous object
    coded_objects, num_objects = sp.ndimage.label(filled)
    print_array(coded_objects)
    np.set_printoptions() # reset print options
    #show_image(coded_objects, 3)

    # Isolate the extent of each object
    data_slices = sp.ndimage.find_objects(coded_objects)
    print("data_slices cnt: " + str(len(data_slices)))

    return data_slices

def read_file(filename):
    """Returns a iterator 
    """
    reader = imageio.get_reader(filename)
    return reader

def read_frame(reader, index):
    """Reads a frame from the reader."""

    im = reader.get_next_data()

    print(im)
    print(type(im))

    n = np.frombuffer(im, dtype="uint8")

    return n
    
    #next = reader.get_next_data()
    #frame_header = reader.next().strip().split()
    #time = float(frame_header[-2][1:])
    #data = []
    #while True:
    #    line = reader.next().strip().split()
    #    if line == []:
    #        break
    #    data.append(line)
    #return time, np.array(data, dtype=np.float)

if __name__ == '__main__':
    video = sys.argv[1]
    animate(video)
#    animate('Grouped up objects.bin')
#    animate('Normal measurement.bin')
