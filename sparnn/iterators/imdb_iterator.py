__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT
import theano.tensor.nnet
import random
import cPickle
from sparnn.utils import *
from sparnn.iterators import PklIterator

logger = logging.getLogger(__name__)


class IMDBIterator(PklIterator):
    def __init__(self, iterator_param):
        super(IMDBIterator, self).__init__(iterator_param)
        self.vocabulary_size = 10000



