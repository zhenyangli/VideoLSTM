__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)


class SamplingLayer(Layer):
    def __init__(self, layer_param):
        super(SamplingLayer, self).__init__(layer_param)
        self.sampling_func = layer_param.get('sampling_func', "Multinomial")
        self.fprop()

    def set_name(self):
        self.name = "SamplingLayer-" + str(self.id)

    def step_fprop(self, input):
        output = quick_sampling(input, self.sampling_func, self.theano_rng)
        return output

    def fprop(self):
        self.output = self.step_fprop(self.input)

