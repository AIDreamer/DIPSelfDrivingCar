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
        self.head = None # No heads have been implemented -_-
        self.losses = []
        self.metrics = []
        self.batch_size = kwargs.get('batch_size', 4) # Defaults to 4
        self.learning_rate = kwargs.get('learning_rate', 0.001) # Defaults to 0.001
        
        # Dataset is tuple of (tf.data.Dataset, tf.data.Iterator)
        # E.g. element = self.dataset[1].get_next() is a tuple of a datum and its label
        self.dataset = kwargs.get('dataset', None)
        # These will be set when self.Train() or self.Run() are called
        self.tensor = None
        self.labels = None
        
        # Graph that holds all of the models datasets, operations, and tensors
        self._graph = tf.Graph()
            
    
            
    def set_dataset(self, data_path, labels_path=''):
        self.dataset = utils.load_dataset(data_path, self._graph, labels_path)
        return
    
    # ---------------------------------------------------------------------------- #
    # Wrappers for tf.nn.xxx
    # ---------------------------------------------------------------------------- #
        
    def Conv(self, filter_size, depth, features, strides=[1,1,1,1], padding='SAME', use_cudnn_on_gpu=None, data_format='NHWC', name=None):
        """
        filter_shape: [filter_height, filter_width, in_channels(depth), out_channels(features)]
        
        can add biases if needed
        """
        filter_shape = [filter_size, filter_size, depth, features]
        weights = tf.get_variable('weights', shape=filter_shape).initialized_value()
        self.tensor = tf.nn.conv2d(self.tensor, weights, strides, padding, use_cudnn_on_gpu, data_format, name)
        return
    
    def ConvTranspose(self, filter_size, features, depth, output_shape, strides=[1,1,1,1], padding='SAME', data_format='NHWC', name=None):
        filter_shape = [filter_size, filter_size, features, depth]
        weights = tf.get_variable('weights', shape=filter_shape).initialized_value()
        self.tensor = tf.nn.conv2d_transpose(self.tensor, weights, output_shape, strides, padding, data_format, name)
        return
    
    def FullyConnected(self):
        pass
    
    def Relu(self, name=None):
        self.tensor = tf.nn.relu(self.tensor, name=name)
        return                           
    
    def MaxPool(self, ksize, strides, padding='SAME', data_format='NHWC', name=None):
        self.tensor = tf.nn.max_pool(self.tensor, ksize, strides, padding)
        return
    
    def StopGradient(self, name=None):
        self.tensor = tf.nn.self_gradient(self.tensor, name=name)
        return
    
    def Softmax(self):
        self.tensor = tf.nn.softmax(self.tensor)
        pass

    def RunBatch(self,):
        """
        Runs the model on a batch.
        """
        
        with self._graph.as_default():
            # Initialize Dataset
            assert self.dataset != None, 'Dataset required to train'
            batched_dataset = self.dataset[0].batch(self.batch_size)
            iterator = batched_dataset.make_initializable_iterator()
            next_element = iterator.get_next()
        
            init = (iterator.make_initializer(batched_dataset))

        with tf.Session(graph=self._graph) as sess:
            sess.run(init)
            try:
                batch = tf.data.Dataset.from_tensor_slices(sess.run(next_element))
                self.tensor, self.labels = utils.stack_batch_into_tensor(batch, self._graph, train=self.train)
            except tf.errors.OutOfRangeError:
                pass
            
            output = self.body(self)
            
        return output
    
    def RunSingle(self, image_filename):
        """
        Runs the model on a test image.
        """
        
        with self._graph.as_default():
            self.tensor = utils.load_image(image_filename, self._graph)
            init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        
        with tf.Session(graph=self._graph) as sess:
            sess.run(init)
            output = self.body(self)
            
        return output
        
    
#     def Train(self):
#         """
#         Main function to train the model on its current dataset.
#         """
#         with self._graph.as_default():
#             # Initialize Dataset
#             assert self.dataset != None, 'Dataset required to train'
#             batched_dataset = self.dataset[0].batch(self.batch_size)
#             iterator = batched_dataset.make_initializable_iterator()
#             next_element = iterator.get_next()

#             init = (iterator.make_initializer(batched_dataset))

#         # Need to initialize self.tensor and self.labels before adding graph operations
#         with tf.Session(graph=self._graph) as sess:
#             sess.run(init)
#             try:
#                 batch = tf.data.Dataset.from_tensor_slices(sess.run(next_element))
#                 self.tensor, self.labels = utils.stack_batch_into_tensor(batch, self._graph, train=self.train)
#                 self.labels = utils.prep_labels(self.labels, self.num_classes, self._graph)
#             except tf.errors.OutOfRangeError:
#                 pass
#         # ---------------------------------------------------------------------------- #
#         # Graph operations and tensors
#         # ---------------------------------------------------------------------------- #
#         with self._graph.as_default():
#             self.body(self)
#             self.Softmax()

#             # Get loss
#             """
#             This might be a problem because the above two lines that execute the network don't
#             have to be run to get loss_fn (since it just uses self.tensor and self.labels which 
#             already exist). might have to put them in variables and have loss_fn use those.
#             """
#             loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.tensor, labels=self.labels))
#             # loss_fn = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.tensor, labels=self.labels))

#             # Optimizer
#             optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss_fn)

#             # Initializer for all variables and iterators
#             init = (tf.global_variables_initializer(), 
#                     tf.local_variables_initializer(), 
#                     iterator.make_initializer(batched_dataset),
#                     self.dataset[1].make_initializer(self.dataset[0]))
        
#         # ---------------------------------------------------------------------------- #
#         # Session
#         # ---------------------------------------------------------------------------- #
#         with tf.Session(graph=self._graph) as sess:
#             sess.run(init)
            
#             # Get batch and divide into data and labels
#             # while True:
#             for i in range(3): # Shorted loop for testing
#                 try:
#                     batch = tf.data.Dataset.from_tensor_slices(sess.run(next_element))
#                     self.tensor, self.labels = utils.stack_batch_into_tensor(batch, self._graph, train=self.train)
#                     self.labels = utils.prep_labels(self.labels, self.num_classes, self._graph)
                    
#                     loss = sess.run(loss_fn) # For printing values while debugging only
#                     sess.run(optimizer)
#                     print('Loss:', loss)
#                 except tf.errors.OutOfRangeError:
#                     break
            
#         return


    def Train(self):
        """
        Main function to train the model on its current dataset.
        """
        output_dir = cfg.MODEL.OUTPUT_DIR
        weights_file = os.path.join(output_dir, cfg.TRAIN.WEIGHTS)
        
        checkpoints = {}
        cur_iter = 0
        
        with self._graph.as_default():
            # Initialize Dataset
            assert self.dataset != None, 'Dataset required to train'
            batched_dataset = self.dataset[0].batch(self.batch_size)
            iterator = batched_dataset.make_initializable_iterator()
            next_element = iterator.get_next()

            init = (iterator.make_initializer(batched_dataset))

        # Need to initialize self.tensor and self.labels before adding graph operations
        with tf.Session(graph=self._graph) as sess:
            sess.run(init)
            try:
                batch = tf.data.Dataset.from_tensor_slices(sess.run(next_element))
                self.tensor, self.labels = utils.stack_batch_into_tensor(batch, self._graph, train=self.train)
                self.labels = utils.prep_labels(self.labels, self.num_classes, self._graph)
            except tf.errors.OutOfRangeError:
                pass
        # ---------------------------------------------------------------------------- #
        # Graph operations and tensors
        # ---------------------------------------------------------------------------- #
        with self._graph.as_default():
                
            self.body(self)
            self.Softmax()

            # Get loss
            """
            This might be a problem because the above two lines that execute the network don't
            have to be run to get loss_fn (since it just uses self.tensor and self.labels which 
            already exist). might have to put them in variables and have loss_fn use those.
            """
            loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.tensor, labels=self.labels))
            # loss_fn = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.tensor, labels=self.labels))

            # Optimizer
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss_fn)

            # Initializer for all variables and iterators
            init = (tf.global_variables_initializer(), 
                    tf.local_variables_initializer(), 
                    iterator.make_initializer(batched_dataset),
                    self.dataset[1].make_initializer(self.dataset[0]))
        
        # ---------------------------------------------------------------------------- #
        # Session
        # ---------------------------------------------------------------------------- #
        with tf.Session(graph=self._graph) as sess:
            sess.run(init)
            saver = tf.train.Saver()
            
            # Get batch and divide into data and labels
            while True:
                try:
                    batch = tf.data.Dataset.from_tensor_slices(sess.run(next_element))
                    self.tensor, self.labels = utils.stack_batch_into_tensor(batch, self._graph, train=self.train)
                    self.labels = utils.prep_labels(self.labels, self.num_classes, self._graph)
                    
                    loss = sess.run(loss_fn) # For printing values while debugging only
                    sess.run(optimizer)
                    print('Loss:', loss)
                    
                    if (cur_iter + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
                        ckpt_num = (cur_iter + 1) // cfg.CHECKPOINT_PERIOD
                        svae_path = weights_file + '_checkpoint_' + str(ckpt_num)
                        checkpoints[ckpt_num] = saver.save(sess, save_path)
                        print("Model saved in path: %s" % save_path)
                    cur_iter += 1
                except tf.errors.OutOfRangeError:
                    break
            
            save_path = weights_file + '_final'
            checkpoints['final'] = saver.save(sess, save_path)
            print("Model saved in path: %s" % save_path)
            
        return




def create_model():
    """
    Creates a new model using the configurations in cfg (from config.py)
    
    Returns:
    - model: A Model object containing the model
    - weights_file: A string containing the path to file storing the weights 
    - output_dir: A string containing the path to the directory of the weights_file
    """
    body = cfg.MODEL.CONV_BODY
    output_dir = cfg.MODEL.OUTPUT_DIR
    if output_dir[-1] != '/':
        output_dir += '/'
    weights_file = os.path.join(output_dir, cfg.TRAIN.WEIGHTS)
    num_classes = cfg.MODEL.NUM_CLASSES
    train = cfg.TRAIN.TRAINING
    batch_size = cfg.TRAIN.BATCH_SIZE
    learning_rate = cfg.TRAIN.LEARNING_RATE
    # implement auto resume 
    model = Model(num_classes=num_classes, body=body, train=train, batch_size=batch_size, learning_rate=learning_rate)
    return model, weights_file, output_dir

                                     
    
                                     
    
    
        
        