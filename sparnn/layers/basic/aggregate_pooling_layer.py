__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT
import theano.tensor.nnet
from sparnn.utils import *
from sparnn.layers import Layer

logger = logging.getLogger(__name__)


class AggregatePoolingLayer(Layer):
    def __init__(self, layer_param):
        super(AggregatePoolingLayer, self).__init__(layer_param)
        self.pooling_func = layer_param["pooling_func"]
        self.fprop()

    def set_name(self):
        self.name = "AggregatePoolingLayer-" + str(self.id)

    def step_fprop(self, input, mask=None):
        output = quick_aggregate_pooling(input, self.pooling_func, mask)
        return output

    def fprop(self):
        self.output = self.step_fprop(self.input, self.mask)
