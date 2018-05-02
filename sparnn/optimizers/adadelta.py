__author__ = 'sxjscience'
import numpy
import time
import theano
import logging
import theano.tensor as TT
from sparnn.utils import *
from sparnn.optimizers import Optimizer
logger = logging.getLogger(__name__)


class AdaDelta(Optimizer):
    """
        Zeiler, Matthew D. "ADADELTA: an adaptive learning rate method." arXiv preprint arXiv:1212.5701 (2012).

    """
    def __init__(self,
                 model,
                 train_data_iterator,
                 valid_data_iterator,
                 test_data_iterator,
                 hyper_param
                 ):
        super(AdaDelta, self).__init__(model, train_data_iterator, valid_data_iterator, test_data_iterator, hyper_param)
        self.decay_rate = numpy_floatX(hyper_param["decay_rate"])

    def set_name(self):
        self.name = "AdaDelta-" + self.id

    def get_update_func(self):
        updates = []
        rho = TT.scalar(self._s("decay_rate"), dtype=theano.config.floatX)
        eps = numpy_floatX(1E-6)
        self.g2_list = [theano.shared(p.get_value() * numpy_floatX(0.), name="%s.g2" % p.name) for p in self.model.param]
        self.dx2_list = [theano.shared(p.get_value() * numpy_floatX(0.), name="%s.dx2e" % p.name) for p in self.model.param]

        updates += [(p, p - TT.sqrt(dx2+eps)/TT.sqrt(rho*g2 + (1-rho)*TT.square(g) + eps)*g)
                    for p, g, g2, dx2 in zip(self.model.param, self.grad, self.g2_list, self.dx2_list)]
        updates += [(dx2, rho*dx2 + (1-rho)*(dx2+eps)/(rho*g2 + (1-rho)*TT.square(g) + eps)*TT.square(g))
                    for g, g2, dx2 in zip(self.grad, self.g2_list, self.dx2_list)]
        updates += [(g2, rho*g2 + (1-rho)*TT.square(g))
                    for g, g2 in zip(self.grad, self.g2_list)]
        return self.model.get_update_func(updates, [rho])

    def learning_param(self):
        return [self.decay_rate]

    def print_stat(self):
        super(AdaDelta, self).print_stat()
        logger.info("   Learning Parameters:")
        logger.info("      Clipping Threshold: " + str(self.clip_threshold))
        logger.info("      Decay Rate: " + str(self.decay_rate))

