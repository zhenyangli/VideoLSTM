__author__ = 'zhenyang'
import numpy
import time
import theano
import logging
import theano.tensor as TT
from sparnn.utils import *
from sparnn.optimizers import Optimizer
logger = logging.getLogger(__name__)


class Adam(Optimizer):
    """
        Kingma et al. "Adam: A Method for Stochastic Optimization" arXiv preprint: http://arxiv.org/abs/1412.6980 (ICLR 2015).
        Theano implementation adapted from Soren Kaae Sonderby (https://github.com/skaae)
        
    """
    def __init__(self,
                 model,
                 train_data_iterator,
                 valid_data_iterator,
                 test_data_iterator,
                 hyper_param
                 ):
        super(Adam, self).__init__(model, train_data_iterator, valid_data_iterator, test_data_iterator, hyper_param)
        self.learning_rate = numpy_floatX(hyper_param["learning_rate"])
        self.beta1 = numpy_floatX(hyper_param["beta1"]) # 0.9
        self.beta2 = numpy_floatX(hyper_param["beta2"]) # 0.999

    def set_name(self):
        self.name = "Adam-" + self.id

    def get_update_func(self):
        updates = []
        lr = TT.scalar(self._s("learning_rate"), dtype=theano.config.floatX)
        b1 = TT.scalar(self._s("beta1"), dtype=theano.config.floatX)
        b2 = TT.scalar(self._s("beta2"), dtype=theano.config.floatX)
        eps = numpy_floatX(1E-8) # 1E-8
        self.time = theano.shared(numpy_floatX(0.), name=self._s("time"))
        t = self.time + 1.
        lr_t = lr * TT.sqrt(1. - b2**t) / (1. - b1**t)

        self.m_list = [theano.shared(p.get_value() * numpy_floatX(0.), name="%s.m" % p.name) for p in self.model.param]
        self.v_list = [theano.shared(p.get_value() * numpy_floatX(0.), name="%s.v" % p.name) for p in self.model.param]
        m_t_list = [(b1 * m) + (1. - b1) * g for g, m in zip(self.grad, self.m_list)]
        v_t_list = [(b2 * v) + (1. - b2) * TT.sqr(g) for g, v in zip(self.grad, self.v_list)]
        updates += [(p, p - lr_t * m_t / (TT.sqrt(v_t) + eps)) for p, m_t, v_t in zip(self.model.param, m_t_list, v_t_list)]
        updates += [(m, m_t) for m, m_t in zip(self.m_list, m_t_list)]
        updates += [(v, v_t) for v, v_t in zip(self.v_list, v_t_list)]
        updates += [(self.time, t)]

        #for p, g in zip(self.model.param, self.grad):
        #    m = theano.shared(p.get_value() * numpy_floatX(0.), name="%s.m" % p.name)
        #    v = theano.shared(p.get_value() * numpy_floatX(0.), name="%s.v" % p.name)
        #    m_t = (b1 * m) + (1. - b1) * g
        #    v_t = (b2 * v) + (1. - b2) * TT.sqr(g)
        #    p_t = p - lr_t * m_t / (TT.sqrt(v_t) + eps)
        #    updates.append((m, m_t))
        #    updates.append((v, v_t))
        #    updates.append((p, p_t))
        #updates.append((self.time, self.time+1.))

        return self.model.get_update_func(updates, [lr, b1, b2])

    def learning_param(self):
        return [self.learning_rate, self.beta1, self.beta2]

    def print_stat(self):
        super(Adam, self).print_stat()
        logger.info("   Learning Parameters:")
        logger.info("      Clipping Threshold: " + str(self.clip_threshold))
        logger.info("      Learning Rate: " + str(self.learning_rate))
        logger.info("      Decay rate for the first moment: " + str(self.beta1))
        logger.info("      Decay rate for the second moment: " + str(self.beta2))

