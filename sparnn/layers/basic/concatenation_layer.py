__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)


class ConcatenationLayer(Layer):
    def __init__(self, layer_param):
        super(ConcatenationLayer, self).__init__(layer_param)
        self.axis = layer_param['axis']
        self.fprop()

    def set_name(self):
        self.name = "ConcatenationLayer-" + str(self.id)

    def step_fprop(self, input):
        output = TT.concatenate(input, axis=self.axis)
        return output

    def fprop(self):
        self.output = self.step_fprop(self.input)

