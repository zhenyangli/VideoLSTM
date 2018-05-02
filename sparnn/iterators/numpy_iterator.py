__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT
import theano.tensor.nnet
import random
from sparnn.utils import *
from sparnn.iterators import DataIterator

logger = logging.getLogger(__name__)


class NumpyIterator(DataIterator):
    def __init__(self, iterator_param):
        super(NumpyIterator, self).__init__(iterator_param)
        self.load(self.path)

    def load(self, path):
        dat = numpy.load(path)
        for key in dat.keys():
            self.data[key] = dat[key]
        self.check_data()
