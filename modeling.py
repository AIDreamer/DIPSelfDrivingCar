from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf
from config import cfg
import conv_bodies
import utils

class Model:
    """
    Generic model class
    """
    def __init__(self, **kwargs):
        self.train = kwargs.get('train', False)
        self.num_classes = kwargs.get('num_classes', -1)
        assert self.num_classes > 0, 'number of classes must be greater than zero'
        self.body = kwargs.get('body', conv_bodies.VGG16)
        self.head = 'Mask'
        self.losses = []
        self.metrics = []
        dataset = kwargs.get(dataset, None)
        if dataset != None:
            self.data, self.labels = utils.load_data(dataset)
            self.tensor = tf.Variable(initial_data=self.data)
        else:
            self.data = self.labels = self.tensor = None
            
            
    def set_dataset(self, dataset):
        self.data, self.labels = utils.load_data(dataset)
        self.tensor = tf.Variable(initial_data=self.data,  )
        
    def Conv(self, filter_shape, strides=[1,1,1,1], padding='SAME', use_cudnn_on_gpu=None, data_format=None, name=None):
        """
        filter_shape: [filter_height, filter_width, in_channels(features), out_channels(depth)]
        
        can add biases if needed
        """
        weights = tf.get_variable('weights', shape=filter_shape)
        self.tensor = tf.nn.conv2d(self.tensor, weights, strides, padding, use_cudnn_on_gpu, data_format, name)
        return
        
    def DilatedConv(self, filters_shape, rate=2, padding='SAME', name=None):
        """ 
        rate must be int32
        """
        weights = tf.get_variable('weights', shape=filter_shape)
        self.tensor = tf.nn.atrous_conv2d(self.tensor, weights, rate, padding)
        return
    
    def Relu(self, name=None):
        self.tensor = tf.nn.relu(self.tensor, name=name)
        return                           
    
    def MaxPool(self, ksize, strides, padding='SAME', data_format='NHWC', name=None):
        self.tensor = tf.nn.max_pool(self.tensor, ksize, strides, padding)
        return
    
    def Softmax(self, ):
        pass
    
    

def create_model():
    """
    Creates a new model using the configurations in cfg
    
    Returns:
    - model: A Model object containing the model
    - weights_file: A string containing the path to file storing the weights
    - start_iter: 
    - checkpoints: 
    - output_dir: A string containing the path to the directory of the weights_file
    """
    start_iter = 0
    checkpoints = {}
    body = cfg.MODEL.CONV_BODY
    output_dir = cfg.MODEL.OUTPUT_DIR
    weights_file = os.path.join(output_dir, cfg.TRAIN.WEIGHTS)
    num_classes = cfg.MODEL.NUM_CLASSES
    train = cfg.TRAIN.TRAINING
    # implement auto resume 
    model = Model(num_classes=num_classes, body=body, train=train)
    return model, weights_file, start_iter, checkpoints, output_dir
                                     
    
                                     
    
    
        
        