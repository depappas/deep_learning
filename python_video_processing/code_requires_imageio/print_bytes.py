import imageio
import sys

video = sys.argv[1]

reader = imageio.get_reader(video)

for i, im in enumerate(reader):
    print(im)
