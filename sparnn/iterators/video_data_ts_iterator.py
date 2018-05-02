__author__ = 'zhenyang'

import numpy
import logging
import theano
import theano.tensor as TT
import theano.tensor.nnet
import random
import h5py
from sparnn.utils import *

logger = logging.getLogger(__name__)

'''
VideoDataTsIterator is the iterator for large video data

1. Data Format

The data it iterates from contains these attributes:

input data:  4-dimensional numpy array, (Frame, FeatureDim, Row, Col)
        or   2-dimensional numpy array, (Frame, FeatureDim)
input label: 1-dimensional numpy array, (Frame,)
        or   2-dimensional numpy array, (Frame, Label) for multi-label output

2. Batch Format

input_batch:  5-dimensional numpy array, (Timestep, Minibatch, FeatureDim, Row, Col)
         or   3-dimensional numpy array, (Timestep, Minibatch, FeatureDim)
output_batch: 2-dimensional numpy array, (Timestep, Minibatch,)
         or   3-dimensional numpy array, (Timestep, Minibatch, Label) for multi-label output

3. About Mask

The VideoDataTsIterator class will automatically generate input/output mask if set the `use_mask` flag.
The mask has 2 dims, (Timestep, Minibatch), all elements are either 0 or 1

'''


class VideoDataTsIterator(object):
    def __init__(self, iterator_param):
        self.name = iterator_param['name']
        self.use_mask = iterator_param.get('use_mask', None)
        self.input_data_type = iterator_param.get('input_data_type', theano.config.floatX)
        self.context_data_type = iterator_param.get('context_data_type', theano.config.floatX)
        self.output_data_type = iterator_param.get('output_data_type', theano.config.floatX)
        self.minibatch_size = iterator_param['minibatch_size']
        self.is_output_multilabel =  iterator_param['is_output_multilabel']
        self.one_hot_label =  iterator_param['one_hot_label']
        
        self.dataset = iterator_param['dataset']
        self.data_file = iterator_param['data_file']
        self.context_file = iterator_param['context_file']
        self.num_frames_file = iterator_param['num_frames_file']
        self.labels_file = iterator_param['labels_file']
        self.vid_name_file = iterator_param['vid_name_file']
        self.dataset_name = iterator_param['dataset_name']

        self.reshape = iterator_param.get('reshape', False)

        self.num_segments = iterator_param['num_segments']
        self.train_sampling = iterator_param['train_sampling']
        self.seq_length = iterator_param['seq_length']
        self.seq_fps = iterator_param['seq_fps']
        self.seq_skip = int(30.0/self.seq_fps)

        self.rng = iterator_param['rng']
        self.frame_rng = iterator_param['frame_rng']

        self.data = {}
        self.context = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []

        self.load()

    def load(self):

        # load data
        # self.data = h5py.File(self.data_file,'r')[self.dataset_name]      # load dataset
        # self.data_dims = self.data.shape[1:]                              # 3D cube
        self.data = self.data_file

        # load context
        # self.context = h5py.File(self.context_file,'r')[self.dataset_name] # load dataset
        # self.context_dims = self.context.shape[1:]                         # 1D vector
        self.context = self.context_file

        # load video labels
        if self.is_output_multilabel:
            init_labels = self.get_map_labels(self.labels_file) # multi class labels for mAP
        else:
            init_labels = self.get_labels(self.labels_file)     # labels
        self.labels = numpy.array(init_labels)

        # load number of frames
        num_frames = []                                         # number of frames in each example
        for line in open(self.num_frames_file):
            num_frames.append(int(line.strip()))
        self.lengths = numpy.array(num_frames)
        
        # load video file names
        video_names = []
        for line in open(self.vid_name_file):
            video_names.append(line.strip())
        self.video_names = video_names

        # verify total number of videos
        assert len(video_names) == len(num_frames) == len(init_labels)
        self.num_videos = len(init_labels)
  
        # set up dataset
        self.dataset_size = self.num_videos
        print 'Dataset size', self.dataset_size

        # data statistics
        self.data_dims = h5py.File('%s/%s.h5' % (self.data,self.video_names[0]), 'r')[self.dataset_name].shape[1:]
        self.context_dims = h5py.File('%s/%s.h5' % (self.context,self.video_names[0]), 'r')[self.dataset_name].shape[1:]
        print 'Data dim', self.data_dims
        print 'Context dim', self.context_dims
        if self.is_output_multilabel:
            self.label_dims = self.labels.shape[1:]
        else:
            self.label_dims = (numpy.unique(self.labels).size,)
        print 'Label dim', self.label_dims

        self.check_data()

    def get_labels(self, filename):
        labels = []
        if filename != '':
            for line in open(filename,'r'):
                labels.append(int(line.strip()))
        return labels

    def get_map_labels(self, filename):
        labels = []
        if filename != '':
            for line in open(filename,'r'):
                labels.append([int(x) for x in line.split(',')])
        return labels

    def check_data(self):
        #assert 2 == self.data.ndim or 4 == self.data.ndim
        return True

    def total(self):
        return self.dataset_size

    def begin(self, do_shuffle=True):
        self.indices = numpy.arange(self.total(), dtype="int32")
        if do_shuffle:
            self.rng.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_size = self.minibatch_size if self.current_position \
                                                         + self.minibatch_size <= self.total() else self.total() - self.current_position
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.current_batch_size]

    def next(self):
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        self.current_batch_size = self.minibatch_size if self.current_position \
                                                         + self.minibatch_size <= self.total() else self.total() - self.current_position
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.current_batch_size]

    def no_batch_left(self):
        if self.current_position >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            # TODO Use Log!
            logger.error(
                "There is no batch left in " + self.name + ". Consider to use iterators.begin() to rescan from " \
                                                           "the beginning of the iterators")
            return None
        input_batch = numpy.zeros(
            (self.seq_length, self.minibatch_size*self.num_segments) + tuple(self.data_dims)).astype(
             self.input_data_type)
        ctx_batch = numpy.zeros(
            (self.seq_length, self.minibatch_size*self.num_segments) + tuple(self.context_dims)).astype(
             self.context_data_type)
        mask = numpy.zeros((self.seq_length, self.minibatch_size*self.num_segments)).astype(
                            theano.config.floatX) if self.use_mask else None
        
        if self.is_output_multilabel:
            output_batch = numpy.zeros((self.seq_length, self.minibatch_size*self.num_segments) + tuple(self.label_dims)).astype(
                                        self.output_data_type)
        elif self.one_hot_label:
            output_batch = numpy.zeros((self.seq_length, self.minibatch_size*self.num_segments)).astype(
                                        self.output_data_type)
        else:
            output_batch = numpy.zeros((self.seq_length, self.minibatch_size*self.num_segments) + tuple(self.label_dims)).astype(
                                        self.output_data_type)

        data = None
        context = None
        for i in xrange(self.current_batch_size):
            # move to current batch/video position
            batch_ind = self.current_batch_indices[i]
            vid_ind = batch_ind
            label = self.labels[batch_ind]
            length = self.lengths[batch_ind]

            # load data for current video
            data = h5py.File('%s/%s.h5' % (self.data,self.video_names[vid_ind]), 'r')[self.dataset_name]
            context = h5py.File('%s/%s.h5' % (self.context,self.video_names[vid_ind]), 'r')[self.dataset_name]
            # check total number of frames in dataset
            assert data.shape[0] == context.shape[0]
            for j in xrange(self.num_segments):
                # sample a segment from current video
                if length >= self.seq_length*self.seq_skip:
                    #avg_length = int(length/self.num_segments)
                    avg_length = int((length - self.seq_length*self.seq_skip + 1.)/self.num_segments)
                    #assert avg_length >= self.seq_length*self.seq_skip
                    if self.train_sampling:
                        #offset = self.frame_rng.randint(avg_length - self.seq_length*self.seq_skip + 1)
                        offset = self.frame_rng.randint(avg_length)
                        start = offset + j*avg_length
                        end = start + self.seq_length*self.seq_skip
                        input_batch[:, i*self.num_segments + j, :] = data[start:end:self.seq_skip, :]
                        ctx_batch[:, i*self.num_segments + j, :] = context[start:end:self.seq_skip, :]
                    else:
                        #start = int((avg_length - self.seq_length*self.seq_skip + 1)/2 + j*avg_length)
                        start = int(avg_length/2. + j*avg_length)
                        end = start + self.seq_length*self.seq_skip
                        input_batch[:, i*self.num_segments + j, :] = data[start:end:self.seq_skip, :]
                        ctx_batch[:, i*self.num_segments + j, :] = context[start:end:self.seq_skip, :]
                else:
                    start = 0
                    n = 1 + int((length-1)/self.seq_skip)
                    input_batch[:n, i*self.num_segments + j, :] = data[start:start+length:self.seq_skip, :]
                    input_batch[n:, i*self.num_segments + j, :] = numpy.tile(input_batch[n-1, i*self.num_segments + j, :],
                                                                            (self.seq_length-n,) + ((1,) * len(self.data_dims)))
                    ctx_batch[:n, i*self.num_segments + j, :] = context[start:start+length:self.seq_skip, :]
                    ctx_batch[n:, i*self.num_segments + j, :] = numpy.tile(ctx_batch[n-1, i*self.num_segments + j, :],
                                                                          (self.seq_length-n,) + ((1,) * len(self.context_dims)))

                if self.is_output_multilabel:
                    output_batch[:, i*self.num_segments + j, :] = numpy.tile(label, (self.seq_length,1))
                elif self.one_hot_label:
                    output_batch[:, i*self.num_segments + j] = numpy.tile(label, (1,self.seq_length))
                else:
                    output_batch[:, i*self.num_segments + j, label] = 1.

        # only for testing, will change in the future
        if self.reshape:
            input_batch = input_batch.reshape([input_batch.shape[0], input_batch.shape[1], 
                                               input_batch.shape[2], input_batch.shape[3]*input_batch.shape[4]])
        #input_batch = input_batch.reshape([input_batch.shape[0], input_batch.shape[1], 49, 1024])
        #input_batch = input_batch.transpose((0,1,3,2))
        
        if self.use_mask:
            mask[:, :self.current_batch_size*self.num_segments] = 1.
        input_batch = input_batch.astype(self.input_data_type)
        ctx_batch = ctx_batch.astype(self.context_data_type)
        output_batch = output_batch.astype(self.output_data_type)

        if self.use_mask:
            return [input_batch, ctx_batch, mask, output_batch]
        else:
            return [input_batch, ctx_batch, output_batch]

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("   Dataset: " + self.dataset)
        logger.info("   Minibatch Size: " + str(self.minibatch_size))
        logger.info("   Use Mask: " + str(self.use_mask))
        logger.info("   Input Data Type: " + str(self.input_data_type))
        logger.info("   Output Data Type: " + str(self.output_data_type))
        logger.info("   Is Output Multi Label: " + str(self.is_output_multilabel))

def main():
    exit()

if __name__ == '__main__':
    main()
