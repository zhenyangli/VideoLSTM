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
class NINLayer(Layer):
    def __init__(self, layer_param):
        super(NINLayer, self).__init__(layer_param)
        self.activation = layer_param["activation"]
        self.bias = layer_param.get("bias", True)

        self.W = quick_init_xavier(self.rng, (self.feature_in, self.feature_out), self._s("W"))
        if self.bias:
            self.b = quick_zero((self.feature_out, ), self._s("b"))

        if self.bias:
            self.param = [self.W, self.b]
        else:
            self.param = [self.W]
        self.fprop()

    def set_name(self):
        self.name = "NINLayer-" + str(self.id)

    def step_fprop(self, input):
        # cf * bc01... = fb01...
        out_r = TT.tensordot(self.W, input, axes=[[0], [1]])
        # input dims to broadcast over
        remaining_dims = range(2, input.ndim)
        # bf01...
        out = out_r.dimshuffle(1, 0, *remaining_dims)

        if self.bias:
            remaining_dims_biases = ['x'] * (input.ndim - 2)  # broadcast
            output = quick_activation(out + self.b.dimshuffle('x', 0, *remaining_dims_biases), self.activation)
        else:
            output = quick_activation(out, self.activation)
        return output

    def fprop(self):
        self.output = self.step_fprop(self.input)
