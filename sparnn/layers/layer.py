__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT
from sparnn.utils import *

logger = logging.getLogger(__name__)


class Layer(object):
    """
        Base Class of Layer
        Input: (Timestamp, Minibatch, FeatureDim) or (Timestamp, Minibatch, FeatureDim, Row, Col)
        Mask: (Timestamp, Minibatch)
        minibatch_size should be a tensor shape argument!
    """
    def __init__(self, layer_param):
        self.rng = layer_param['rng']
        self.theano_rng = layer_param['theano_rng']
        self.dim_in = layer_param.get('dim_in', None)
        self.dim_out = layer_param.get('dim_out', None)
        self.input = layer_param.get('input', None)
        self.mask = layer_param.get('mask', None)
        self.minibatch_size = layer_param['minibatch_size']
        self.id = str(layer_param['id'])
        #assert len(self.dim_in) == 3 and len(self.dim_out) == 3
        self.feature_in = self.dim_in[0]
        self.feature_out = self.dim_out[0]
        self.param = []
        self.output = None
        self.output_update = None
        self.is_recurrent = False
        self.set_name()
        self.is_train = theano.shared(numpy_floatX(1.), name=self._s("is_train"))

    def set_mode(self, mode):
        if mode == "train":
            self.is_train.set_value(numpy_floatX(1.))
        elif mode == "predict":
            self.is_train.set_value(numpy_floatX(0.))
        else:
            assert False

    def set_name(self):
        self.name = "BaseLayer-" + str(self.id)

    def total_param_num(self):
        ret = 0
        if len(self.param) >0:
            for p in self.param:
                ret += p.get_value(borrow=True).size
        return ret

    def print_stat(self):
        logger.info(self.name + ":")
        if len(self.param) > 0:
            for p in self.param:
                logger.info("   " + p.name + " : " + str(p.get_value(borrow=True).shape) + " " + str(p.get_value(borrow=True).size))
            logger.info("   Total Parameter Number: " + str(self.total_param_num()))
        else:
            logger.info("   No parameter")

    def _s(self, s):
        return '%s.%s' % (self.name, s)
