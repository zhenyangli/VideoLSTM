__author__ = 'sxjscience'
import numpy
import logging
import theano
import theano.tensor as TT

from sparnn.utils import *
from sparnn.layers import Layer


logger = logging.getLogger(__name__)


class MixtureDensityOutputLayer(Layer):
    def __init__(self, layer_param):
        super(MixtureDensityOutputLayer, self).__init__(layer_param)

    def set_name(self):
        self.name = "MixtureDensityOutputLayer-" + str(self.id)