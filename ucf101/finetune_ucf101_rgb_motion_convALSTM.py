__author__ = 'zhenyang'

import theano
import theano.tensor as TT

import sparnn
import sparnn.utils
from sparnn.utils import *

from sparnn.iterators import VideoDataTsIterator
from sparnn.layers import StackInterfaceLayer
from sparnn.layers import DenseLayer
from sparnn.layers import ConvLayer
from sparnn.layers import DeepCondConvLSTMLayer
from sparnn.layers import DropoutLayer
from sparnn.layers import PredictionLayer
from sparnn.layers import ElementwiseCostLayer

from sparnn.models import VideoModel

from sparnn.optimizers import SGD
from sparnn.optimizers import RMSProp
from sparnn.optimizers import AdaDelta
from sparnn.optimizers import Adam

import os
import random
import numpy


load_path = "./ucf101-experiment/ucf101-rgb-motion-convALSTM/rms-lr-0.001-drop-0.7/"
save_path = "./ucf101-experiment/finetune-ucf101-rgb-motion-convALSTM/rms-lr-0.001-drop-0.7/"

log_path = save_path + "finetune_ucf101_rgb_motion_convALSTM.log"

if not os.path.exists(save_path):
    os.makedirs(save_path)

sparnn.utils.quick_logging_config(log_path)

random.seed(1000)
numpy.random.seed(1000)

iterator_rng = sparnn.utils.quick_npy_rng(1234)
iterator_frame_rng = sparnn.utils.quick_npy_rng(1337)
seq_length = 30

#############################
iterator_param = {'dataset': 'ucf101',
                  'data_file': '/ssd/zhenyang/data/UCF101/features/rgb_vgg16_pool5',
                  'context_file': '/ssd/zhenyang/data/UCF101/features/flow_vgg16_pool5',
                  'num_frames_file': '/ssd/zhenyang/data/UCF101/train_framenum.txt',
                  'labels_file': '/ssd/zhenyang/data/UCF101/train_labels.txt',
                  'vid_name_file': '/ssd/zhenyang/data/UCF101/train_filenames.txt',
                  'dataset_name': 'features', 'rng': iterator_rng, 'frame_rng': iterator_frame_rng,
                  'seq_length': seq_length, 'num_segments': 1, 'seq_fps': 30,
                  'minibatch_size': 128, 'train_sampling': True,
                  'use_mask': True, 'input_data_type': 'float32', 'context_data_type': 'float32',
                  'output_data_type': 'int64', 'one_hot_label': True,
                  'is_output_multilabel': False,
                  'name': 'ucf101-train-video-ts-iterator'}
train_iterator = VideoDataTsIterator(iterator_param)
train_iterator.begin(do_shuffle=True)
train_iterator.print_stat()
#
iterator_param = {'dataset': 'ucf101',
                  'data_file': '/ssd/zhenyang/data/UCF101/features/rgb_vgg16_pool5',
                  'context_file': '/ssd/zhenyang/data/UCF101/features/flow_vgg16_pool5',
                  'num_frames_file': '/ssd/zhenyang/data/UCF101/test_framenum.txt',
                  'labels_file': '/ssd/zhenyang/data/UCF101/test_labels.txt',
                  'vid_name_file': '/ssd/zhenyang/data/UCF101/test_filenames.txt',
                  'dataset_name': 'features', 'rng': None, 'frame_rng': None,
                  'seq_length': seq_length, 'num_segments': 25, 'seq_fps': 30,
                  'minibatch_size': 20, 'train_sampling': False,
                  'use_mask': True, 'input_data_type': 'float32', 'context_data_type': 'float32',
                  'output_data_type': 'int64', 'one_hot_label': True,
                  'is_output_multilabel': False,
                  'name': 'ucf101-valid-video-ts-iterator'}
valid_iterator = VideoDataTsIterator(iterator_param)
valid_iterator.begin(do_shuffle=False)
valid_iterator.print_stat()
#
test_iterator = None

#############################
rng = sparnn.utils.quick_npy_rng()
theano_rng = sparnn.utils.quick_theano_rng(rng)

############################# load model
model_file = load_path + "UCF101-VideoModel-RGB-Motion-convALSTM-RMS-initialization.pkl"
model = VideoModel.load(model_file)
model.print_stat()

############################# optimizer
param = {'id': '1', 'learning_rate': 0.001, 'momentum': 0.9, 'decay_rate': 0.9, 'clip_threshold': None, 'verbose': False,
         'max_epoch': 400, 'start_epoch': 0, 'valid_epoch': 10, 'max_epochs_no_best': 400, 'decay_step': 400,
         'display_freq': 150, 'valid_freq': None, 'save_freq': None,
         'autosave_mode': ['interval', 'best'], 'save_path': save_path, 'save_interval': 10}
optimizer = RMSProp(model, train_iterator, valid_iterator, test_iterator, param)
#param = {'id': '1', 'learning_rate': 0.001, 'momentum': 0.9, 'decay_rate': 0.1, 'clip_threshold': 50, 'verbose': False,
#         'max_epoch': 300, 'start_epoch': 0, 'valid_epoch': 20, 'max_epochs_no_best': 300, 'decay_step': 100,
#         'display_freq': 150, 'valid_freq': None, 'save_freq': None,
#         'autosave_mode': ['interval', 'best'], 'save_path': save_path, 'save_interval': 20}
#optimizer = SGD(model, train_iterator, valid_iterator, test_iterator, param)

optimizer.train()
