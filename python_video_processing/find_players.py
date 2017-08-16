import sys
import numpy as np
import scipy as sp
import scipy.ndimage

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def animate(input_filename):
    """Detects paws  animates the position and raw data of each frame
    in the input file"""
    # With matplotlib, it's much, much faster to just update the properties
    # of a display object than it is to create a new one, so we'll just update
    # the data and position of the same objects throughout this animation...

    infile = paw_file(input_filename)

    # Since we're making an animation with matplotlib, we need 
    # ion() instead of show()...
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle(input_filename)

    # Make an image based on the first frame that we'll update later
    # (The first frame is never actually displayed)
    im = ax.imshow(infile.next()[1])

    # Make 4 rectangles that we can later move to the position of each paw
    rects = [Rectangle((0,0), 1,1, fc='none', ec='red') for i in range(4)]
    [ax.add_patch(rect) for rect in rects]

    title = ax.set_title('Time 0.0 ms')

    # Process and display each frame
    for time, frame in infile:
        paw_slices = find_paws(frame)

        # Hide any rectangles that might be visible
        [rect.set_visible(False) for rect in rects]

        # Set the position and size of a rectangle for each paw and display it
        for slice, rect in zip(paw_slices, rects):
            dy, dx = slice
            rect.set_xy((dx.start, dy.start))
            rect.set_width(dx.stop - dx.start + 1)
            rect.set_height(dy.stop - dy.start + 1)
            rect.set_visible(True)

        # Update the image data and title of the plot
        title.set_text('Time %0.2f ms' % time)
        im.set_data(frame)
        im.set_clim([frame.min(), frame.max()])
        fig.canvas.draw()

def find_paws(data, smooth_radius=5, threshold=0.0001):
    """Detects and isolates contiguous regions in the input array"""
    # Blur the input data a bit so the paws have a continous footprint 
    data = sp.ndimage.uniform_filter(data, smooth_radius)
    # Threshold the blurred data (this needs to be a bit > 0 due to the blur)
    thresh = data > threshold
    # Fill any interior holes in the paws to get cleaner regions...
    filled = sp.ndimage.morphology.binary_fill_holes(thresh)
    # Label each contiguous paw
    coded_paws, num_paws = sp.ndimage.label(filled)
    # Isolate the extent of each paw
    data_slices = sp.ndimage.find_objects(coded_paws)
    return data_slices

def paw_file(filename):
    """Returns a iterator that yields the time and data in each frame
    The infile is an ascii file of timesteps formatted similar to this:

    Frame 0 (0.00 ms)
    0.0 0.0 0.0
    0.0 0.0 0.0

    Frame 1 (0.53 ms)
    0.0 0.0 0.0
    0.0 0.0 0.0
    ...
    """
    with open(filename) as infile:
        i = 0
        while True:
            try:
                print("i: " + str(i))
                time, data = read_frame(infile)
                yield time, data
            except StopIteration:
                break

def read_frame(infile):
    """Reads a frame from the infile."""
    frame_header = infile.next().strip().split()
    time = float(frame_header[-2][1:])
    data = []
    while True:
        line = infile.next().strip().split()
        if line == []:
            break
        data.append(line)
    return time, np.array(data, dtype=np.float)

if __name__ == '__main__':
    video = sys.argv[1]
    animate(video)
#    animate('Grouped up paws.bin')
#    animate('Normal measurement.bin')
