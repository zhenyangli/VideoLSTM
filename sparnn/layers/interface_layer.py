__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT
from sparnn.utils import *

logger = logging.getLogger(__name__)


class InterfaceLayer(object):
    def __init__(self, layer_param):
        self.id = layer_param['id']
        self.use_mask = layer_param.get('use_mask', False)
        input_data_type = layer_param.get('input_data_type', theano.config.floatX)
        input_ndim = layer_param.get('input_ndim', 5)
        output_data_type = layer_param.get('output_data_type', theano.config.floatX)
        output_ndim = layer_param.get('output_ndim', 5)
        self.set_name()
        self.input = quick_symbolic_variable(ndim=input_ndim, name=self._s("input"), typ=input_data_type)
        self.output = quick_symbolic_variable(ndim=output_ndim, name=self._s("output"), typ=output_data_type)
        if self.use_mask:
            self.mask = quick_symbolic_variable(ndim=2, name=self._s("mask"), typ=theano.config.floatX)
        else:
            self.mask = None

    def set_name(self):
        self.name = "InterfaceLayer-" + str(self.id)

    def _s(self, s):
        return '%s.%s' % (self.name, s)

    def mask_symbols(self):
        return [self.mask]

    def input_symbols(self):
        return [self.input]

    def output_symbols(self):
        return [self.output]

    def symbols(self):
        if self.use_mask:
            return self.input_symbols() + self.mask_symbols() + self.output_symbols()
        else:
            return self.input_symbols() + self.output_symbols()

    def print_stat(self):
        logger.info(self.name + ":")
        logger.info("   Use mask: " + str(self.use_mask))
        if self.use_mask:
            logger.debug("  Mask Type:" + str(self.mask.type) + " Mask Name: " + self.mask.name)
        logger.info("   Input Type: " + str(self.input.type) + " Input Name: " + self.input.name)
        logger.info("   Output Type: " + str(self.output.type) + " Output Name: " + self.output.name)
