__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT
from sparnn.utils import *
from sparnn.layers import Layer

logger = logging.getLogger(__name__)


class EmbeddingLayer(Layer):
    def __init__(self, layer_param):
        super(EmbeddingLayer, self).__init__(layer_param)
        self.Wemb = quick_init_gaussian(self.rng, (self.feature_in, self.feature_out), self._s("Wemb"))
        self.param = [self.Wemb]
        self.fprop()

    def set_name(self):
        self.name = "EmbeddingLayer-" + str(self.id)

    def step_fprop(self, input):
        output = TT.shape_padright(self.Wemb[input.flatten()].reshape(
            [input.shape[0], input.shape[1], self.feature_out]), 2)
        return output

    def fprop(self):
        self.output = self.step_fprop(self.input)
