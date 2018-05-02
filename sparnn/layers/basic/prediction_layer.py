__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)


class PredictionLayer(Layer):
    def __init__(self, layer_param):
        super(PredictionLayer, self).__init__(layer_param)
        #assert 3 == self.input.ndim
        self.last_n = layer_param['last_n']
        self.is_multilabel = layer_param['is_multilabel']
        self.fprop()

    def set_name(self):
        self.name = "PredictionLayer-" + str(self.id)

    def step_fprop(self, input):
        if 3 == self.input.ndim:
            if self.is_multilabel:
                output = TT.mean(input[-self.last_n:, :, :], axis=0)    # (TS,BS,#actions) -> (BS,#actions)
            else:
                output = TT.sum(input[-self.last_n:, :, :], axis=0)     # (TS,BS,#actions) -> (BS,#actions)
                output = TT.argmax(output, axis=1) # compute prediction #                  -> (BS,)
        elif 2 == self.input.ndim:
            if self.is_multilabel:
                output = input    # (BS,#actions) -> (BS,#actions)
            else:
                output = TT.argmax(input, axis=1) # compute prediction  # (BS,#actions) -> (BS,)
        else:
            assert False

        return output

    def fprop(self):
        self.output = self.step_fprop(self.input)

