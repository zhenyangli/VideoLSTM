__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)

'''
    Network-in-network layer.
    Like feed-forward layer, but broadcasting across all trailing dimensions beyond the
    2nd.  This results in a convolution operation with filter size 1 on all
    trailing dimensions.
'''
class DenseLayer(Layer):
    def __init__(self, layer_param):
        super(DenseLayer, self).__init__(layer_param)
        self.activation = layer_param["activation"]
        self.bias = layer_param.get("bias", True)

        num_inputs = int(numpy.prod(self.dim_in[:]))
        self.W = quick_init_xavier(self.rng, (num_inputs, self.feature_out), self._s("W"))
        if self.bias:
            self.b = quick_zero((self.feature_out, ), self._s("b"))

        if self.bias:
            self.param = [self.W, self.b]
        else:
            self.param = [self.W]
        self.fprop()

    def set_name(self):
        self.name = "DenseLayer-" + str(self.id)

    def step_fprop(self, input):
        if len(self.dim_in) > 1:
            # if the input has more than one dimension, flatten it into a
            # batch of feature vectors.
            remaining_dims = input.ndim - len(self.dim_in) + 1
            input = input.flatten(remaining_dims)

        out = TT.dot(input, self.W)

        if self.bias:
            output = quick_activation(out + self.b, self.activation)
        else:
            output = quick_activation(out, self.activation)
        return output

    def fprop(self):
        self.output = self.step_fprop(self.input)
