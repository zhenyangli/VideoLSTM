__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT
from sparnn.utils import *

logger = logging.getLogger(__name__)


class Gate(object):
    def __init__(self, layer_param):
        self.id = layer_param['id']
        self.activation = layer_param['activation']
        self.set_name()

        self.W_x = quick_init_norm(self.rng, (self.feature_in, self.feature_out), self._s("W_x"), scale=0.1)
        self.W_h = quick_init_norm(self.rng, (self.feature_in, self.feature_out), self._s("W_h"), scale=0.1)
        self.b = quick_zero((self.feature_out, ), self._s("b"))

        self.param = [self.W_x, self.W_h, self.b]

    def set_name(self):
        self.name = "Gate-" + str(self.id)

    def _s(self, s):
        return '%s.%s' % (self.name, s)
