__author__ = 'zhenyang'

import theano
import theano.tensor as TT

import sys
sys.path.append('../')
import sparnn
import sparnn.utils
from sparnn.utils import *

from sparnn.iterators import VideoDataIterator
from sparnn.layers import InterfaceLayer
from sparnn.layers import FeedForwardLayer
from sparnn.layers import CondLSTMLayer
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


save_path = "./ucf101-experiment/ucf101-flow-ALSTM/rms-lr-0.001-drop-0.5/"
log_path = save_path + "evaluate_ucf101_flow_ALSTM.log"

if not os.path.exists(save_path):
    os.makedirs(save_path)

sparnn.utils.quick_logging_config(log_path)

random.seed(1000)
numpy.random.seed(1000)

iterator_rng = sparnn.utils.quick_npy_rng(1337)
iterator_frame_rng = sparnn.utils.quick_npy_rng(1234)
seq_length = 30

#############################
iterator_param = {'dataset': 'ucf101',
                  'data_file': '/ssd/zhenyang/data/UCF101/features/flow_vgg16_pool5',
                  'num_frames_file': '/ssd/zhenyang/data/UCF101/train_framenum.txt',
                  'labels_file': '/ssd/zhenyang/data/UCF101/train_labels.txt',
                  'vid_name_file': '/ssd/zhenyang/data/UCF101/train_filenames.txt',
                  'dataset_name': 'features', 'rng': iterator_rng, 'frame_rng': iterator_frame_rng,
                  'seq_length': seq_length, 'num_segments': 1, 'seq_fps': 30,
                  'minibatch_size': 128, 'train_sampling': True, 'reshape': True,
                  'use_mask': True, 'input_data_type': 'float32', 'output_data_type': 'int64', 'one_hot_label': True,
                  'is_output_multilabel': False,
                  'name': 'ucf101-train-video-iterator'}
train_iterator = VideoDataIterator(iterator_param)
train_iterator.begin(do_shuffle=True)
train_iterator.print_stat()
#
iterator_param = {'dataset': 'ucf101',
                  'data_file': '/ssd/zhenyang/data/UCF101/features/flow_vgg16_pool5',
                  'num_frames_file': '/ssd/zhenyang/data/UCF101/test_framenum.txt',
                  'labels_file': '/ssd/zhenyang/data/UCF101/test_labels.txt',
                  'vid_name_file': '/ssd/zhenyang/data/UCF101/test_filenames.txt',
                  'dataset_name': 'features', 'rng': None, 'frame_rng': None,
                  'seq_length': seq_length, 'num_segments': 25, 'seq_fps': 30,
                  'minibatch_size': 20, 'train_sampling': False, 'reshape': True,
                  'use_mask': True, 'input_data_type': 'float32', 'output_data_type': 'int64', 'one_hot_label': True,
                  'is_output_multilabel': False,
                  'name': 'ucf101-valid-video-iterator'}
valid_iterator = VideoDataIterator(iterator_param)
valid_iterator.begin(do_shuffle=False)
valid_iterator.print_stat()
#
test_iterator = None

#############################
rng = sparnn.utils.quick_npy_rng()
theano_rng = sparnn.utils.quick_theano_rng(rng)

############################# interface layer
param = {"id": "ucf101-flow-vgg16-pool5", "use_mask": True,
         "input_ndim": 4, "output_ndim": 2,
         "output_data_type": "int64"}
interface_layer = InterfaceLayer(param)

x = interface_layer.input
mask = interface_layer.mask
y = interface_layer.output

timesteps = x.shape[0]
minibatch_size = x.shape[1]

feature_dim = 512
hidden_dim = 512
out_dim = 1024
regions = 7*7
actions = 101
data_dim = (feature_dim, regions)

logger.info("Data Dim:" + str(data_dim))

# initial state/cell (Timestep, Minibatch, FeatureDim, Region)
input_mean = x.mean(0) ### input_mean is now (Minibatch, FeatureDim, Region)
input_mean = input_mean.mean(2) ### you want input_mean to be Minibatch x FeatureDim


#############################
middle_layers = []

#0# initalization layer for lstm state
param = {"id": 0, "rng": rng, "theano_rng": theano_rng,
          "dim_in": (feature_dim,), "dim_out": (hidden_dim,),
          "minibatch_size": minibatch_size,
          "activation": "tanh",
          "input": input_mean}
middle_layers.append(FeedForwardLayer(param))

#1# initalization layer for lstm memory
param = {"id": 1, "rng": rng, "theano_rng": theano_rng,
          "dim_in": (feature_dim,), "dim_out": (hidden_dim,),
          "minibatch_size": minibatch_size,
          "activation": "tanh",
          "input": input_mean}
middle_layers.append(FeedForwardLayer(param))

#2# conditional lstm layer (main layer)
param = {"id": 2, "rng": rng, "theano_rng": theano_rng,
         "dim_in": data_dim, "dim_out": (hidden_dim,),
         "minibatch_size": minibatch_size,
         "input": x, "mask": mask,
         "init_hidden_state": middle_layers[0].output,
         "init_cell_state": middle_layers[1].output,
         "temperature_inverse": 1.,
         "n_steps": seq_length}
middle_layers.append(CondLSTMLayer(param))

#3# set up dropout 1
param = {"id": 3, "rng": rng, "theano_rng": theano_rng,
          "dim_in": (hidden_dim,), "dim_out": (hidden_dim,),
          "minibatch_size": minibatch_size,
          "dropout_rate": 0.5,
          "input": middle_layers[2].output}
middle_layers.append(DropoutLayer(param))

#4# output layer
param = {"id": 4, "rng": rng, "theano_rng": theano_rng,
          "dim_in": (hidden_dim,), "dim_out": (out_dim,),
          "minibatch_size": minibatch_size,
          "activation": "tanh",
          "input": middle_layers[3].output}
middle_layers.append(FeedForwardLayer(param))

#5# set up dropout 2
param = {"id": 5, "rng": rng, "theano_rng": theano_rng,
          "dim_in": (out_dim,), "dim_out": (out_dim,),
          "minibatch_size": minibatch_size,
          "dropout_rate": 0.5,
          "input": middle_layers[4].output}
middle_layers.append(DropoutLayer(param))

#6# classification layer (softmax outputs class probabilities)
param = {"id": 6, "rng": rng, "theano_rng": theano_rng,
          "dim_in": (out_dim,), "dim_out": (actions,),
          "minibatch_size": minibatch_size,
          "activation": "softmax",
          "input": middle_layers[5].output}
middle_layers.append(FeedForwardLayer(param))

#7# label prediction layer
#param = {"id": 7, "rng": rng, "theano_rng": theano_rng,
#         "dim_in": (actions,), "dim_out": (1,),
#         "minibatch_size": minibatch_size,
#         "last_n": seq_length,
#         "is_multilabel": False,
#         "input": middle_layers[6].output}
#middle_layers.append(PredictionLayer(param))

############################# cost layer
param = {"id": "cost", "rng": rng, "theano_rng": theano_rng,
         "dim_in": (actions,), "dim_out": (1,),
         "minibatch_size": minibatch_size,
         "cost_func": "CategoricalCrossEntropy",
         #"regularization": "l2",
         "param_layers": middle_layers,
         #"penalty_rate": 0.00001,
         "input": middle_layers[6].output,
         "mask": mask,
         "target": y}
cost_layer = ElementwiseCostLayer(param)

outputs = [{"name": "probability", "value": middle_layers[6].output}]
# error_layers = [cost_layer]

############################# model
param = {'interface_layer': interface_layer, 'middle_layers': middle_layers, 'cost_layer': cost_layer,
         'outputs': outputs, 'errors': None, 'last_n': seq_length,
         'name': "UCF101-VideoModel-Flow-ALSTM-RMS",
         'problem_type': "classification"}
model = VideoModel(param)
model.print_stat()

############################# optimizer
param = {'id': '1', 'learning_rate': 0.001, 'decay_rate': 0.9, 'clip_threshold': None, 'verbose': False,
         'max_epoch': 200, 'start_epoch': 0, 'valid_epoch': 20, 'max_epochs_no_best': 200,
         'display_freq': 150, 'valid_freq': None, 'save_freq': None,
         'autosave_mode': ['interval', 'best'], 'save_path': save_path, 'save_interval': 20}
optimizer = RMSProp(model, train_iterator, valid_iterator, test_iterator, param)


optimizer.train()
