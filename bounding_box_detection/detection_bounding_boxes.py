import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob
import errno


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#def ensure_dir(file_path):
#    directory = os.path.dirname(file_path)
#    if not os.path.exists(directory):
#        os.makedirs(directory)

def mkdir(dir):
  try:
    os.makedirs(dir)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
      
def test_dir(dir):
  if (not os.path.isdir(dir)):
    print("ERROR: " + dir + " directory does not exist: " + str(dir))
    exit()

def is_file(file):
  if (not os.path.exists(file)):
    print("ERROR: " + file + " does not exist: " + file)
    exit()

# Env setup

# This is needed to display the images.
#%matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")

# Object detection imports
# Here are the imports from the object detection module.

#from utils import label_map_util
# https://github.com/tensorflow/models/issues/1591
# ipython notebook example file is in "object_detection" folder.
# So , print(os.getcwd()) will show you "/models/object_detection"
# but , "from object_detection.utils import label_map_util"
# this line load utils from "/models"
# current directory "/models/object_detection" but utils recognize current directory as "/models"
# So, one trick is work.
# "from object_detection.utils import label_map_util" --> "from utils import label_map_util"
# And it works.

from object_detection.utils import label_map_util

#from utils import visualization_utils as vis_util
from object_detection.utils import visualization_utils as vis_util

# Model preparation
# Variables
# 
# Any model exported using the export_inference_graph.py tool can be
# loaded here simply by changing PATH_TO_CKPT to point to a new .pb file.
# 
# By default we use an "SSD with Mobilenet" model here. See the
# detection model zoo for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'inception_v4_2016_09_09'

MODEL_FILE = MODEL_NAME + '.tar.gz'

DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.

tensorFLowModelDir = os.environ['TENSORFLOW_MODELS_DIR']

test_dir(tensorFLowModelDir)

mscoco_label_map_file = tensorFLowModelDir+ '/object_detection/data/mscoco_label_map.pbtxt'

is_file(mscoco_label_map_file)
    
PATH_TO_LABELS = os.path.join('data', mscoco_label_map_file)

NUM_CLASSES = 90

# Download Model

opener = urllib.request.URLopener()

opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

tar_file = tarfile.open(MODEL_FILE)

for file in tar_file.getmembers():

  file_name = os.path.basename(file.name)

  if 'frozen_inference_graph.pb' in file_name:

    tar_file.extract(file, os.getcwd())

# Load a (frozen) Tensorflow model into memory.

print("Detection graph: Load a (frozen) Tensorflow model into memory.")

detection_graph = tf.Graph()

with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map

# Label maps map indices to category names, so that when our convolution network predicts 5,
# we know that this corresponds to airplane. Here we use internal
# utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

print("Loading label map")

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

category_index = label_map_util.create_category_index(categories)

# Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Detection

#PATH_TO_TEST_IMAGES_DIR = '/home/depappas/Dropbox/soccer_images/rm_mc_2017'
PATH_TO_TEST_IMAGES_DIR = '/home/depappas/Dropbox/soccer_images/bcn_rm_test'
#PATH_TO_TEST_IMAGES_DIR = '/home/depappas/Dropbox/soccer_images/example'
PATH_TO_TEST_IMAGES_DIR = '/home/depappas/Dropbox/soccer_images/bcn_psg_6_1'

print("PATH_TO_TEST_IMAGES_DIR: " + PATH_TO_TEST_IMAGES_DIR)

test_dir(PATH_TO_TEST_IMAGES_DIR)
    
TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR + "/*png")
TEST_IMAGE_PATHS += glob.glob(PATH_TO_TEST_IMAGES_DIR + "/*jpg")
TEST_IMAGE_PATHS.sort()

# Size, in inches, of the output images.

IMAGE_SIZE = (12, 8)

export_dir = PATH_TO_TEST_IMAGES_DIR + "/images_bb"

mkdir(export_dir)

test_dir(export_dir)

def print_tf_obj(s, obj):
  np.set_printoptions(precision=3)  
  print(s + " = " + str(obj))

  
#  print("boxes = " + tf.Print(b))  
#  for box in boxes:
#    print("Image[" + str(i) + "] box["+ str(b) + "] = " + tf.Print(box))  
#    b = b + 1

print("Detection")

i = 0

with detection_graph.as_default():
  print("detection_graph.as_default():")
  with tf.Session(graph=detection_graph) as sess:
    print("with tf.Session(graph=detection_graph) as sess:")
    print("TEST_IMAGE_PATHS: : " + str(TEST_IMAGE_PATHS))
    for image_path in TEST_IMAGE_PATHS:
      is_file(image_path)
      #if (not os.path.exists(image_path)):
      #  print("ERROR: Image file does not exist: " + str(image_path))
      #  exit()

      print("processing image #: " + str(i) + " file: " + image_path)
      i = i +1
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)

      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
         
      # Each score represents the level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      # Visualization of the results of a detection.

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(image_np)

      s = '%015d' % i
      export_path = os.path.join(export_dir, 'export-' + str(s) + '.png')

      print("export_path: " + export_path)
      print("image_np: " + str(image_np.shape))
      vis_util.save_image_array_as_png(image_np, export_path)

      print_tf_obj("boxes ", boxes)
      print_tf_obj("scores ", scores)
      print_tf_obj("classes ", classes)
