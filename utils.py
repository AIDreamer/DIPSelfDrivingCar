from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------- #
# See import_data_examples.ipynb for code and documentation on this
# ---------------------------------------------------------------------------- #

# Square image size for training
IMAGE_SIZE = 224

def load_dataset(data_path, labels_path=''): 
    """
    Loads a complete dataset. The parse functions cannot use the more general
    tf.image.image_decode function because the output tensor doens't have shape
    which is required for tf.image.resize_images.
    
    Input:
    - data_path: A string containing a path the the data directory
    - label_path: A string containing a path to the labels directory. If this
      is an empty string, there are no labels (i.e. testing)
    
    Returns:
    - dataset: A tf.data.Dataset object containing data and labels
    - iterator: A tf.data.Iterator object for dataset
    """
    
    def _parse_with_labels(data_file, label_file):
        data_string = tf.read_file(data_file)
        label_string = tf.read_file(label_file)
        data = tf.image.decode_jpeg(data_string)
        label = tf.image.decode_jpeg(label_string)
        data_resized = tf.image.resize_images(data, [IMAGE_SIZE, IMAGE_SIZE])
        label_resized = tf.image.resize_images(label, [IMAGE_SIZE, IMAGE_SIZE])
        return data_resized, label_resized
    
    def _parse_without_labels(data_file):
        data_string = tf.read_file(data_file)
        data = tf.image.decode_jpeg(data_string)
        data_resized = tf.image.resize_images(data, [IMAGE_SIZE, IMAGE_SIZE])
        return data_resized

    # Get data
    data_filenames = tf.train.match_filenames_once(data_path + '*.jpg').initialized_value()
    
    if labels_path:
        labels_filenames = tf.train.match_filenames_once(labels_path + '*.png').initialized_value()
    
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((data_filenames, 
                                                      labels_filenames))
        dataset = dataset.map(_parse_with_labels)
    else:
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(data_filenames)
        dataset = dataset.map(_parse_without_labels)
    
    # Create iterator
    iterator = dataset.make_initializable_iterator()
    
    return dataset, iterator

# def load_image(filename):
#     def _parse(filename):
#         image_string = tf.read_file(filename)
#         image_tensor = tf.image.decode_jpeg(image_string) # Single images do not have to be resized
#         return image_tensor
        
#     dataset = tf.data.Dataset.from_tensor_slices((filename))
#     dataset = dataset.map(_parse)
    
#     # Create iterator
#     iterator = dataset.make_initializable_iterator()
    
#     return dataset, iterator
    
def load_image(filename):
    image_string = tf.read_file(filename)
    image_tensor = tf.image.decode_jpeg(image_string) # Single images do not have to be resized
    
    # This is weird
    # tf.image.decode_image outputs a tensor with partial shape (?, ?, ?). We need a tensor of known 
    # shape, so we have to run the tensor to get a numpy array with the shape, then convert that 
    # array back to a tensor.
    with tf.Session() as sess:
        image = sess.run(image_tensor)
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32) # dtype cannot be uint8 (default)
        # The tensor needs a batch size for then network.
        # tf.expand_dims inserts a dimension of 1 along the specifiec axis.
        image_tensor = tf.expand_dims(image_tensor, 0)

    return image_tensor
    

# ---------------------------------------------------------------------------- #
# See model_testing.ipynb for code and documentation on this
# ---------------------------------------------------------------------------- #

def stack_batch_into_tensor(batch, train=False):
    """
    Inputs:
    - batch: A tf.data.Dataset object. Elements of this dataset may have shape
      (1,) which will be (data_tensor) if train is False or (2,) which will be 
      (data_tensor, labels_tensor) if train is true.
    - train: A boolean that specifies whether the batch is for training. This 
      will effect the shape of the elements in batch due to labels included for
      training.
      
    Returns:
    - 
    """
    
    def _stack_with_labels(batch):
        """
        Function to handle a batch where elements have both data and label tensors
        """
        iterator = batch.make_initializable_iterator()
        next_element = iterator.get_next()
        init = iterator.make_initializer(batch)

        with tf.Session() as sess:

            # Initialize the iterator on the training data
            sess.run(init)

            element = sess.run(next_element)
            data_tensor_list = [element[0]]
            label_tensor_list = [element[1]]

            while True:
                try:
                    elem = sess.run(next_element)
                    data_tensor_list += [elem[0]]
                    label_tensor_list += [elem[1]]
                except tf.errors.OutOfRangeError:
                    break
                    
            data_tensor = tf.stack(data_tensor_list, axis=0)
            label_tensor = tf.stack(label_tensor_list, axis=0)

        return data_tensor, label_tensor
        
    def _stack_without_labels(batch):
        """
        Function to handle a batch where elements have only data tensors
        """
        iterator = batch.make_initializable_iterator()
        next_element = iterator.get_next()
        init = iterator.make_initializer(batch)

        with tf.Session() as sess:

            # Initialize the iterator on the training data
            sess.run(init)

            tensor_list = [sess.run(next_element)]

            while True:
                try:
                    tensor_list += [sess.run(next_element)]
                except tf.errors.OutOfRangeError:
                    break
            tensor = tf.stack(tensor_list, axis=0)

        return tensor

    if train:
        return _stack_with_labels(batch)
    else:
        return _stack_without_labels(batch)
    
    


   

