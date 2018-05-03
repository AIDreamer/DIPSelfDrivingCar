from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

def VGG16(model):
    """
    The base VGG16 network. Note that the softmax layer and last fully connected
    layer are removed, and the remaining two fully connected layers have been 
    converted to convolutions, effectively making it an FCN. This is meant as an 
    extensible base network and is not meant to be used without adjustments. 
    
    Inputs:
    - model: A Model object with a Tensor member that will be sent through the network
    
    Returns:
    -output: A tf.Tensor object repressenting the output of the network
    """
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
        model.Conv(3, 3, 64, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
        model.Conv(3, 64, 64, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
        model.Conv(3, 64, 128, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE):
        model.Conv(3, 128, 128, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE):
        model.Conv(3, 128, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv6', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv7', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv8', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv9', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv10', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv11', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv12', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv13', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv14', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 4096, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv15', reuse=tf.AUTO_REUSE):
        model.Conv(3, 4096, 4096, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    return model.tensor

        
def FCN_32(model):
    """
    Implements the FCN-32 network. Takes the VGG16 network as base, adds a 1x1
    convolution at the end with num_classes features to predict class scores, and
    then upsamples to the size of the original image (but where each pixel has a 
    channel depth of num_classes). The upsample is done in a single transposed
    convolution (upsamples be a factor of 32x). 
    
    Inputs:
    - model: A Model object with a Tensor member that will be sent through the network
    
    Returns:
    -output: A tf.Tensor object repressenting the output of the network
    """
    original_shape = model.tensor.get_shape().as_list()
    final_shape = original_shape[:3] + [model.num_classes+1]
    print('Starting network', model.tensor.shape)
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
        model.Conv(3, 3, 64, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    print('Layer 1 done', model.tensor.shape)
    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
        model.Conv(3, 64, 64, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
        model.Conv(3, 64, 128, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE):
        model.Conv(3, 128, 128, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE):
        model.Conv(3, 128, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv6', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv7', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv8', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv9', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv10', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv11', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv12', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv13', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv14', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 4096, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv15', reuse=tf.AUTO_REUSE):
        model.Conv(3, 4096, 4096, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        
    # Get Class Scores
    with tf.variable_scope('conv16', reuse=tf.AUTO_REUSE):
        model.Conv(1, 4096, model.num_classes+1, padding='SAME', strides=[1,1,1,1])
        
    # Upsample
    with tf.variable_scope('deconv1', reuse=tf.AUTO_REUSE):
        model.ConvTranspose(3, model.num_classes+1, model.num_classes+1, final_shape, padding='SAME', strides=[1,32,32,1])
        model.Relu()
    return model.tensor

def FCN_8(model):
    """
    Implements the FCN-8 network. Takes the VGG16 network as base. After pool3, pool4,
    and conv7, "skips" are added that compute class scores with a 1x1 convolution and 
    then upsample to the shape of pool3. These three tensors are then added and upsampled
    again (by a factor of 8x).
    
    Inputs:
    - model: A Model object with a Tensor member that will be sent through the network
    
    Returns:
    -output: A tf.Tensor object repressenting the output of the network
    """
    original_shape = model.tensor.get_shape().as_list()
    final_shape = original_shape[:3] + [model.num_classes+1]
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
        model.Conv(3, 3, 64, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
        model.Conv(3, 64, 64, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
        model.Conv(3, 64, 128, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE):
        model.Conv(3, 128, 128, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE):
        model.Conv(3, 128, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv6', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv7', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
        pool3 = model.tensor
        
    # Get scores for pool3
    with tf.variable_scope('conv16', reuse=tf.AUTO_REUSE):
        filter_shape = [1, 1, 256, model.num_classes+1]
        weights = tf.get_variable('weights', shape=filter_shape).initialized_value()
        pool3 = tf.nn.conv2d(pool3, weights, padding='SAME', strides=[1,1,1,1])
        
    with tf.variable_scope('conv8', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv9', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv10', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
        pool4 = model.tensor
        
    # Get scores for pool4
    with tf.variable_scope('conv17', reuse=tf.AUTO_REUSE):
        filter_shape = [1, 1, 512, model.num_classes+1]
        weights = tf.get_variable('weights', shape=filter_shape).initialized_value()
        pool4 = tf.nn.conv2d(pool4, weights, padding='SAME', strides=[1,1,1,1])
        
    with tf.variable_scope('conv11', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv12', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv13', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv14', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 4096, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv15', reuse=tf.AUTO_REUSE):
        model.Conv(3, 4096, 4096, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        conv7 = model.tensor
        
    # Get scores for conv7
    with tf.variable_scope('conv18', reuse=tf.AUTO_REUSE):
        filter_shape = [1, 1, 4096, model.num_classes+1]
        weights = tf.get_variable('weights', shape=filter_shape).initialized_value()
        conv7 = tf.nn.conv2d(conv7, weights, padding='SAME', strides=[1,1,1,1])
    
    # Upsample coarse outputs
    with tf.variable_scope('deconv1', reuse=tf.AUTO_REUSE):
        filter_shape = [3, 3, model.num_classes+1, model.num_classes+1]
        weights = tf.get_variable('weights', shape=filter_shape).initialized_value()
        pool4_2x = tf.nn.conv2d_transpose(pool4, weights, pool3.shape, padding='SAME', strides=[1,2,2,1])
    with tf.variable_scope('deconv2', reuse=tf.AUTO_REUSE):
        filter_shape = [3, 3, model.num_classes+1, model.num_classes+1]
        weights = tf.get_variable('weights', shape=filter_shape).initialized_value()
        conv7_4x = tf.nn.conv2d_transpose(conv7, weights, pool3.shape, padding='SAME', strides=[1,4,4,1])
    
    # Add Tensors
    model.tensor = tf.add(pool3, pool4_2x)
    model.tensor = tf.add(model.tensor, conv7_4x)
    
    # Final Upsampling
    with tf.variable_scope('deconv3', reuse=tf.AUTO_REUSE):
        filter_shape = [3, 3, model.num_classes+1, model.num_classes+1]
        weights = tf.get_variable('weights', shape=filter_shape).initialized_value()
        model.tensor = tf.nn.conv2d_transpose(model.tensor, weights, final_shape, padding='SAME', strides=[1,8,8,1])
        
    return model.tensor

def DilatedNet(model):
    """
    Implements the DilatedNet network. Takes the VGG16 as base, and replaces
    pool3 and pool4 with dilated convolutions with dilatinos of 2 and 4 
    respectively. Adds a 1x1 convolution with num_classes depth at the end to 
    predict classes and then upsamples to the original image size in a single 
    step (32x).Honestly not entirely sure if this is the correct implementation
    but it runs okay. 
    
    Inputs:
    - model: A Model object with a Tensor member that will be sent through the network
    
    Returns:
    -output: A tf.Tensor object repressenting the output of the network
    """
    original_shape = model.tensor.get_shape().as_list()
    final_shape = original_shape[:3] + [model.num_classes+1]
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
        model.Conv(3, 3, 64, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
        model.Conv(3, 64, 64, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
        model.Conv(3, 64, 128, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv4', reuse=tf.AUTO_REUSE):
        model.Conv(3, 128, 128, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv5', reuse=tf.AUTO_REUSE):
        model.Conv(3, 128, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv6', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv7', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 256, padding='SAME', strides=[1,1,1,1])
        model.Relu()

        # atrous conv
    with tf.variable_scope('aconv1', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 256, padding='SAME', strides=[1,2,2,1])
        model.Relu()

    with tf.variable_scope('conv8', reuse=tf.AUTO_REUSE):
        model.Conv(3, 256, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv9', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv10', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()

        # atrous conv
    with tf.variable_scope('aconv2', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,4,4,1])
        model.Relu()

    with tf.variable_scope('conv11', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv12', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv13', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 512, padding='SAME', strides=[1,1,1,1])
        model.Relu()
        model.MaxPool(ksize=[1,1,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv14', reuse=tf.AUTO_REUSE):
        model.Conv(3, 512, 4096, padding='SAME', strides=[1,1,1,1])
        model.Relu()
    with tf.variable_scope('conv15', reuse=tf.AUTO_REUSE):
        model.Conv(3, 4096, 4096, padding='SAME', strides=[1,1,1,1])
        model.Relu()

    # Get Class Scores
    with tf.variable_scope('conv16', reuse=tf.AUTO_REUSE):
        model.Conv(1, 4096, model.num_classes+1, padding='SAME', strides=[1,1,1,1])

    # Upsample
    with tf.variable_scope('deconv1', reuse=tf.AUTO_REUSE):
        model.ConvTranspose(3, model.num_classes+1, model.num_classes+1, final_shape, padding='SAME', strides=[1,32,32,1])
        model.Relu()
    return model.tensor