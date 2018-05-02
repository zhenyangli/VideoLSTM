__author__ = 'sxjscience'
import numpy
import time
import theano
import logging
import theano.tensor as TT
from sparnn.utils import *
from sparnn.optimizers import Optimizer
logger = logging.getLogger(__name__)


class AdaGrad(Optimizer):
    """
        Duchi, J., Hazan, E., & Singer, Y. "Adaptive subgradient methods for online learning and stochastic optimization"
        Chris Dyer "Notes on AdaGrad." http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf.

    """
    def __init__(self,
                 model,
                 train_data_iterator,
                 valid_data_iterator,
                 test_data_iterator,
                 hyper_param
                 ):
        super(AdaGrad, self).__init__(model, train_data_iterator, valid_data_iterator, test_data_iterator, hyper_param)
        self.learning_rate = numpy_floatX(hyper_param["learning_rate"])

    def set_name(self):
        self.name = "AdaGrad-" + self.id

    def get_update_func(self):
        updates = []
        lr = TT.scalar(self._s("learning_rate"), dtype=theano.config.floatX)
        eps = numpy_floatX(1E-6)
        self.g2_list = [theano.shared(p.get_value() * numpy_floatX(0.), name="%s.acc_g" % p.name) for p in self.model.param]
        g2_new_list = [g2 + TT.square(g) for g, g2 in zip(self.grad, self.g2_list)]
        updates += [(g2, g2_new) for g2, g2_new in zip(self.g2_list, g2_new_list)]
        updates += [(p, p - lr*g/TT.sqrt(g2_new + eps)) for p, g, g2_new in zip(self.model.param, self.grad, g2_new_list)]
        return self.model.get_update_func(updates, [lr])

    def learning_param(self):
        return [self.learning_rate]

    def print_stat(self):
        super(AdaGrad, self).print_stat()
        logger.info("   Learning Parameters:")
        logger.info("      Clipping Threshold: " + str(self.clip_threshold))
        logger.info("      Learning Rate: " + str(self.decay_rate))

