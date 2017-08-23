import pylab
import imageio
import sys

filename = sys.argv[1]

vid = imageio.get_reader(filename,  'ffmpeg')
nums = [10000, 15000]
for num in nums:
    image = vid.get_data(num)
    fig = pylab.figure()
    fig.suptitle('image #{}'.format(num), fontsize=20)
    pylab.imshow(image)
pylab.show()
