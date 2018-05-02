__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)


class ConvForwardLayer(Layer):
    def __init__(self, layer_param):
        super(ConvForwardLayer, self).__init__(layer_param)
        assert self.input.ndim == 5 or self.input.ndim == 4
        self.input_padding = layer_param.get('input_padding', None)
        self.conv_type = layer_param.get('conv_type', "valid")
        self.input_receptive_field = layer_param['input_receptive_field']
        self.input_stride = layer_param['input_stride']
        self.activation = layer_param['activation']
        self.kernel_size = (
            self.feature_out, self.feature_in, self.input_receptive_field[0], self.input_receptive_field[1])
        self.W_xo = quick_init_xavier(self.rng, self.kernel_size, self._s("W_xo"))
        self.b_o = quick_zero((self.feature_out, ), self._s("b_o"))

        self.param = [self.W_xo, self.b_o]
        self.fprop()

    def set_name(self):
        self.name = "ConvForwardLayer-" + str(self.id)

    def step_fprop(self, input):
        if "valid" == self.conv_type:
            if 5 == input.ndim:
                reshape_input = input.reshape((input.shape[0] * input.shape[1], input.shape[2],
                                               input.shape[3], input.shape[4]))
                # TODO This "use_multiply" utility can be wrapped in sparnn.util. Similar situations occur in ConvLSTM/ConvRNN
                output = quick_conv2d(input=reshape_input, filters=self.W_xo, border_mode='valid',
                                      subsample=self.input_stride, image_shape=(None,) + self.dim_in,
                                      filter_shape=self.kernel_size) + self.b_o.dimshuffle('x', 0, 'x', 'x')
                output = output.reshape((input.shape[0], input.shape[1]) + self.dim_out)
                output = quick_activation(output, self.activation)
            elif 4 == input.ndim:
                output = quick_conv2d(input=input, filters=self.W_xo, border_mode='valid',
                                      subsample=self.input_stride, image_shape=(None,) + self.dim_in,
                                      filter_shape=self.kernel_size) + self.b_o.dimshuffle('x', 0, 'x', 'x')
                output = quick_activation(output, self.activation)
            else:
                assert False
            return output
        elif "same" == self.conv_type:
            if 5 == input.ndim:
                reshape_input = input.reshape((input.shape[0] * input.shape[1], input.shape[2],
                                               input.shape[3], input.shape[4]))
                output = conv2d_same(input=reshape_input, filters=self.W_xo, input_shape=(None,) + self.dim_in,
                                     filter_shape=self.kernel_size) + self.b_o.dimshuffle('x', 0, 'x', 'x')
                output = output.reshape((input.shape[0], input.shape[1]) + self.dim_out)
                output = quick_activation(output, self.activation)
            elif 4 == input.ndim:
                output = conv2d_same(input=input, filters=self.W_xo, input_shape=(None,) + self.dim_in,
                                     filter_shape=self.kernel_size) + self.b_o.dimshuffle('x', 0, 'x', 'x')
                output = quick_activation(output, self.activation)
            else:
                assert False
            return output

    def fprop(self):
        self.output = self.step_fprop(self.input)
