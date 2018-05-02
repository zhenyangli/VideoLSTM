__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer

logger = logging.getLogger(__name__)

'''
    The dropout layer is a regularizer that randomly sets input values to
    zero. dropout_rate is the probability of setting a value to zero.
'''
class DropoutLayer(Layer):
    def __init__(self, layer_param):
        super(DropoutLayer, self).__init__(layer_param)
        self.dropout_rate = layer_param['dropout_rate']
        assert 0 <= self.dropout_rate <= 0.9
        self.fprop()

    def set_name(self):
        self.name = "DropoutLayer-" + str(self.id)

    def step_fprop(self, input):
        output = TT.switch(self.is_train,
                           input * self.theano_rng.binomial(size=input.shape, p=1 - self.dropout_rate, n=1,
                                                            dtype=theano.config.floatX) / (1 - self.dropout_rate),
                           input)
        return output

    def fprop(self):
        self.output = self.step_fprop(self.input)

