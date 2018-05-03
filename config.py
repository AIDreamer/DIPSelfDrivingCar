from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import conv_bodies


class ConfigDict(dict):
    
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)
            
    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value
            
            
cfg = ConfigDict()

# ---------------------------------------------------------------------------- #
# Model cofigurations
# ---------------------------------------------------------------------------- #

cfg.MODEL = ConfigDict()

cfg.MODEL.OUTPUT_DIR = 'models/'

cfg.MODEL.CONV_BODY = conv_bodies.FCN_8 # function object

cfg.MODEL.NUM_CLASSES = 150

# ---------------------------------------------------------------------------- #
# Train cofigurations
# ---------------------------------------------------------------------------- #

cfg.TRAIN = ConfigDict()

cfg.TRAIN.TRAINING = True

cfg.TRAIN.WEIGHTS = 'b' # weight filename

cfg.TRAIN.BATCH_SIZE = 4
cfg.TRAIN.LEARNING_RATE = 0.005

cfg.TRAIN.CHECKPOINT_PERIOD = 1000


        
        