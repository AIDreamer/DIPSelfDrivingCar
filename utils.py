from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

# https://www.tensorflow.org/programmers_guide/datasets
# https://gist.github.com/eerwitt/518b0c9564e500b4b50f
def load_dataset(data_path): 
    """
    Input:
    - dataset: A string containing a path the the directory of a dataset
    
    Returns:
    - data: A numpy array containing the data
    - labels: A numpuy array containinf the labels
    """
    # Make a queue of file names including all the JPEG images files in the relative
    # image directory.
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(data_path + "*.jpg"))

    # Read an entire image file which is required since they're JPEGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    _, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image = tf.image.decode_jpeg(image_file)
    
# ---------------------------------------------------------------------------- #
# Other way
# ---------------------------------------------------------------------------- #

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

# ---------------------------------------------------------------------------- #
# Yet another way
# ---------------------------------------------------------------------------- #

filename_queue = tf.train.string_input_producer(['/Users/HANEL/Desktop/tf.png']) #  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)

  # Start populating the filename queue.

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1): #length of your filename list
    image = my_img.eval() #here is your image Tensor :) 

  print(image.shape)
  Image.fromarray(np.asarray(image)).show()

  coord.request_stop()
  coord.join(threads)