__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT
import theano.tensor.nnet
import random
import cPickle
from sparnn.utils import *
from sparnn.iterators import DataIterator

logger = logging.getLogger(__name__)


class PklIterator(DataIterator):
    def __init__(self, iterator_param):
        super(PklIterator, self).__init__(iterator_param)
        self.load(self.path)

    def load(self, path):
        f = open(path, 'rb')
        dat = cPickle.load(f)
        f.close()
        assert len(dat['input']) == len(dat['output'])
        total_input_timestep = sum(len(l) for l in dat['input'])
        total_output_timestep = sum(len(l) for l in dat['output'])
        self.data['dims'] = numpy.asarray((dat['input_dim'], dat['output_dim']), dtype="int32")
        self.data['input_raw_data'] = numpy.zeros((total_input_timestep,) + dat['input_dim'])
        self.data['output_raw_data'] = numpy.zeros((total_output_timestep,) + dat['output_dim'])
        self.data['clips'] = numpy.zeros((2, len(dat['input']), 2), dtype="int32")
        pos = 0
        for i in range(len(dat['input'])):
            self.data['input_raw_data'][pos:(pos + dat['input'][i].shape[0]), :, :, :] = dat['input'][i]
            self.data['clips'][0, i, 0] = pos
            self.data['clips'][0, i, 1] = dat['input'][i].shape[0]
            pos += dat['input'][i].shape[0]
        pos = 0
        for i in range(len(dat['output'])):
            self.data['output_raw_data'][pos:(pos + dat['output'][i].shape[0]), :, :, :] = dat['output'][i]
            self.data['clips'][1, i, 0] = pos
            self.data['clips'][1, i, 1] = dat['output'][i].shape[0]
            pos += dat['output'][i].shape[0]
        self.check_data()

    def save(self, path):
        numpy.savez(path, dims=self.data['dims'], input_raw_data=self.data['input_raw_data'],
                    output_raw_data=self.data['output_raw_data'], clip=self.data['clip'])