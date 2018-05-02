__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)


class FeedForwardLayer(Layer):
    def __init__(self, layer_param):
        super(FeedForwardLayer, self).__init__(layer_param)
        self.activation = layer_param['activation']
        
        self.W = quick_init_norm(self.rng, (self.feature_in, self.feature_out), self._s("W"), scale=0.1)
        #self.W = quick_init_he_norm(self.rng, (self.feature_in, self.feature_out), self._s("W"))
        self.b = quick_zero((self.feature_out, ), self._s("b"))

        self.param = [self.W, self.b]
        self.fprop()

    def set_name(self):
        self.name = "FeedForwardLayer-" + str(self.id)

    def step_fprop(self, input):
        output = quick_activation(TT.dot(input, self.W) + self.b, self.activation)
        return output

    def fprop(self):
        self.output = self.step_fprop(self.input)
