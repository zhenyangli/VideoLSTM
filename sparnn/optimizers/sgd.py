__author__ = 'sxjscience'
import numpy
import time
import theano
import logging
import theano.tensor as TT
import cPickle
from sparnn.utils import *
from sparnn.optimizers import Optimizer

logger = logging.getLogger(__name__)


class SGD(Optimizer):
    """
        First Order Stochastic Gradient Descent With Momentum
        The clipping strategy is the same as the ICML 2013 paper:On the difficulty of training recurrent neural networks
    """

    def __init__(self,
                 model,
                 train_data_iterator,
                 valid_data_iterator,
                 test_data_iterator,
                 hyper_param
                 ):
        super(SGD, self).__init__(model, train_data_iterator, valid_data_iterator, test_data_iterator, hyper_param)
        self.learning_rate = hyper_param["learning_rate"]
        self.momentum = hyper_param["momentum"]
        self.decay_rate = hyper_param.get("decay_rate", numpy_floatX(0.1))
        self.decay_step = hyper_param.get("decay_step", (self.max_epoch - self.start_epoch) / 3 + 1)
        self.decay_begin = hyper_param.get("decay_begin", 0)

    def set_name(self):
        self.name = "SGD-" + self.id

    def get_update_func(self):
        updates = []
        lr = TT.scalar(self._s("learning_rate"), dtype=theano.config.floatX)
        momentum = TT.scalar(self._s("SGD.momentum"), dtype=theano.config.floatX)
        self.grad_last_update = [theano.shared(p.get_value() * numpy_floatX(0.), name="%s.grad_last_update" % p.name)
                                 for p in self.model.param]
        updates += [(p, p + momentum * p_last_update - lr * p_grad)
                    for p, p_grad, p_last_update in zip(self.model.param, self.grad, self.grad_last_update)]
        updates += [(p_last_update, momentum * p_last_update - lr * p_grad)
                    for p_grad, p_last_update in zip(self.grad, self.grad_last_update)]
        return self.model.get_update_func(updates, [lr, momentum])

    def learning_param(self):
        #if (0 == (self.current_epoch - self.start_epoch + 1) % self.decay_step) and (
        #            self.current_epoch - self.start_epoch) > self.decay_begin:
        if (0 == (self.current_epoch - self.start_epoch) % self.decay_step) and (
                    self.current_epoch - self.start_epoch) >= self.decay_begin:
            self.learning_rate *= self.decay_rate
        return [self.learning_rate, self.momentum]

    def print_stat(self):
        super(SGD, self).print_stat()
        logger.info("   Learning Parameters:")
        logger.info("      Learning Rate: " + str(self.learning_rate))
        logger.info("      Momentum: " + str(self.momentum))
        logger.info("      Decay Rate: " + str(self.decay_rate))
        logger.info("      Decay Step: " + str(self.decay_step))
    