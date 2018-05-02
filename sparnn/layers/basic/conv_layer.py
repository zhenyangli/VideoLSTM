__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)


class ConvLayer(Layer):
    def __init__(self, layer_param):
        super(ConvLayer, self).__init__(layer_param)
        self.receptive_field = layer_param['receptive_field']
        self.activation = layer_param['activation']
        self.input_padding = layer_param.get('input_padding', None)
        
        self.kernel_size = (self.feature_out, self.feature_in,
                            self.receptive_field[0], self.receptive_field[1])

        self.W = quick_init_xavier(self.rng, self.kernel_size, self._s("W"))
        self.b = quick_zero((self.feature_out, ), self._s("b"))

        self.param = [self.W, self.b]
        self.fprop()

    def set_name(self):
        self.name = "ConvLayer-" + str(self.id)

    def step_fprop(self, input):
        output = quick_activation(conv2d_same(input, self.W, (None, ) + self.dim_in,
                                              self.kernel_size, self.input_padding)
                                  + self.b.dimshuffle('x', 0, 'x', 'x'), self.activation)
        return output

    def fprop(self):
        self.output = self.step_fprop(self.input)
