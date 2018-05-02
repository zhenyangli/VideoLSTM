__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)


class ReshapeLayer(Layer):
    def __init__(self, layer_param):
        super(ReshapeLayer, self).__init__(layer_param)
        self.fprop()

    def set_name(self):
        self.name = "ReshapeLayer-" + str(self.id)

    def step_fprop(self, input):
        output = input.reshape(self.dim_out)
        return output

    def fprop(self):
        self.output = self.step_fprop(self.input)

