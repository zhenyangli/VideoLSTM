__author__ = 'sxjscience'

import numpy
import logging
import theano
import theano.tensor as TT
import theano.tensor.nnet
import random
from sparnn.utils import *

logger = logging.getLogger(__name__)

'''
DataIterator is basically the iterator for clip data

1. Data Format

The data it iterates from contains these attributes:

dims: numpy array with the shape (2,3) if the output/input has different dimension, or (1,3) if they have same dimension
clips: 3-dimensional numpy int32 array with shape 2*n*2 --> (Input/Output, Instance, StartPoint/Length),
       StartPoint begin at 0

*** If dims has shape(2,3) ***
input_raw_data: 4-dimensional numpy array, (Timestep, FeatureDim, Row, Col)
output_raw_data: 4-dimensional numpy array, (Timestep, FeatureDim, Row, Col)

*** Else ***
input_raw_data: 4-dimensional numpy array, (Timestep, FeatureDim, Row, Col)

2. About Mask

The DataIterator class will automatically generate input/output mask if set the `use_input_mask` or `use_output_mask` flag.
The mask has 2 dims, (Timestep, Minibatch), all elements are either 0 or 1

'''


class DataIterator(object):
    def __init__(self, iterator_param):
        self.path = iterator_param['path']
        self.use_input_mask = iterator_param.get('use_input_mask', None)
        self.use_output_mask = iterator_param.get('use_output_mask', None)
        self.name = iterator_param['name']
        self.input_data_type = iterator_param.get('input_data_type', theano.config.floatX)
        self.output_data_type = iterator_param.get('output_data_type', theano.config.floatX)
        self.minibatch_size = iterator_param['minibatch_size']
        self.is_output_sequence = iterator_param['is_output_sequence']
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0

    def check_data(self):
        assert 3 == self.data['clips'].ndim
        assert self.data['clips'].dtype == numpy.dtype("int32")
        assert 2 == self.data['dims'].ndim or 1 == self.data['dims'].ndim
        if 1 == self.data['dims'].ndim:
            self.data['dims'] = self.data['dims'].reshape((1, 3))
        assert self.data['dims'].dtype == numpy.dtype("int32")
        assert self.data['dims'].shape == (2, 3) or self.data['dims'].shape == (1, 3)
        assert self.data['input_raw_data'].ndim == 4
        assert self.data['input_raw_data'].shape[1:4] == tuple(self.data['dims'][0])
        if self.data['dims'].shape == (2, 3):
            assert self.data['output_raw_data'].shape[1:4] == tuple(self.data['dims'][1])
            assert self.data['output_raw_data'].ndim == 4

    def total(self):
        return self.data["clips"].shape[1]

    def begin(self, do_shuffle=True):
        self.indices = numpy.arange(self.total(), dtype="int32")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_size = self.minibatch_size if self.current_position \
                                                         + self.minibatch_size <= self.total() else self.total() - self.current_position
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.current_batch_size]
        self.current_input_length = max(self.data['clips'][0, ind, 1] for ind in self.current_batch_indices)
        self.current_output_length = max(self.data['clips'][1, ind, 1] for ind in self.current_batch_indices)

    def next(self):
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        self.current_batch_size = self.minibatch_size if self.current_position \
                                                         + self.minibatch_size <= self.total() else self.total() - self.current_position
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.current_batch_size]
        self.current_input_length = max(self.data['clips'][0, ind, 1] for ind in self.current_batch_indices)
        self.current_output_length = max(self.data['clips'][1, ind, 1] for ind in self.current_batch_indices)

    def no_batch_left(self):
        if self.current_position >= self.total():
            return True
        else:
            return False

    def input_batch(self):
        if self.no_batch_left():
            # TODO Use Log!
            logger.error(
                "There is no batch left in " + self.name + ". Consider to use iterators.begin() to rescan from " \
                                                           "the beginning of the iterators")
            return None
        input_batch = numpy.zeros(
            (self.current_input_length, self.current_batch_size) + tuple(self.data['dims'][0])).astype(
            self.input_data_type)
        input_mask = numpy.zeros((self.current_input_length, self.current_batch_size)).astype(
            theano.config.floatX) if self.use_input_mask else None
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            begin = self.data['clips'][0, batch_ind, 0]
            end = self.data['clips'][0, batch_ind, 0] + self.data['clips'][0, batch_ind, 1]
            data_slice = self.data['input_raw_data'][begin:end, :, :, :]
            if self.use_input_mask:
                input_batch[:data_slice.shape[0], i, :, :, :] = data_slice
                input_mask[:data_slice.shape[0], i] = 1
            else:
                input_batch[:self.current_input_length, i, :, :, :] = data_slice
        input_batch = input_batch.astype(self.input_data_type)
        if self.use_input_mask:
            return [input_batch, input_mask]
        else:
            return [input_batch]

    def output_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to use iterators.begin() to rescan from " \
                                                           "the beginning of the iterators")
            return None
        raw_dat = self.data['output_raw_data'] if (2, 3) == self.data['dims'].shape else self.data['input_raw_data']
        if self.is_output_sequence:
            output_dim = self.data['dims'][0] if (1, 3) == self.data['dims'].shape else self.data['dims'][1]
            output_batch = numpy.zeros((self.current_output_length, self.current_batch_size)
                                       + tuple(output_dim))
            output_mask = numpy.zeros(
                (self.current_output_length, self.current_batch_size)).astype(theano.config.floatX) \
                if self.use_output_mask else None
        else:
            output_batch = numpy.zeros((self.current_batch_size, ) + tuple(self.data['dims'][1]))
            output_mask = None
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            begin = self.data['clips'][1, batch_ind, 0]
            end = self.data['clips'][1, batch_ind, 0] + self.data['clips'][1, batch_ind, 1]
            if self.is_output_sequence:
                data_slice = raw_dat[begin:end, :, :, :]
                output_batch[:data_slice.shape[0], i, :, :, :] = data_slice
                if self.use_output_mask:
                    output_mask[:data_slice.shape[0], i] = 1
            else:
                assert 1 == end - begin and self.use_output_mask is not True
                data_slice = raw_dat[begin, :, :, :]
                output_batch[i, :, :, :] = data_slice
        output_batch = output_batch.astype(self.output_data_type)
        if self.use_output_mask:
            return [output_batch, output_mask]
        else:
            return [output_batch]

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("   Path: " + self.path)
        logger.info("   Minibatch Size: " + str(self.minibatch_size))
        logger.info("   Input Data Type: " + str(self.input_data_type) + " Use Input Mask: " + str(self.use_input_mask))
        logger.info("   Output Data Type: " + str(self.output_data_type) + " Use Output Mask: " + str(self.use_output_mask))
        logger.info("   Is Output Sequence: " + str(self.is_output_sequence))
