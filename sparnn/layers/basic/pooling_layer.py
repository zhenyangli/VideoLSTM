__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)

'''
    Global pooling layer.
    This layer pools globally across all trailing dimensions beyond the 3nd.
'''
class PoolingLayer(Layer):
    def __init__(self, layer_param):
        super(PoolingLayer,self).__init__(layer_param)
        assert 5 == self.input.ndim
        self.pooling_func = layer_param["pooling_func"]
        self.fprop()

    def set_name(self):
        self.name = "PoolingLayer-" + str(self.id)

    def step_fprop(self, input):
        if self.pooling_func == "max":
            return input.flatten(4).max(axis=3)
        elif self.pooling_func == "mean":
            return input.flatten(4).mean(axis=3)
        else:
            return None

    def fprop(self):
        self.output = self.step_fprop(self.input)
