
# coding: utf-8

# In[1]:


import theano
import theano.tensor as TT

import sparnn
import sparnn.utils
from sparnn.utils import *

from sparnn.layers import InterfaceLayer
from sparnn.layers import StackInterfaceLayer
from sparnn.layers import DenseLayer
from sparnn.layers import ConvLayer
from sparnn.layers import ConvLSTMLayer
from sparnn.layers import CondConvLSTMLayer
from sparnn.layers import DeepCondConvLSTMLayer
from sparnn.layers import DropoutLayer
from sparnn.layers import PredictionLayer
from sparnn.layers import ElementwiseCostLayer

from sparnn.models import VideoModel

import os
import random
import numpy


# In[2]:

############################# model config
src1_model_file = "../ucf101-experiment/ucf101-flow-convLSTM/rms-lr-0.001-drop-0.7/UCF101-VideoModel-Flow-convLSTM-RMS-validation-best.pkl"
src2_model_file = "../ucf101-experiment/ucf101-rgb-convALSTM/rms-lr-0.001-drop-0.7/UCF101-VideoModel-RGB-convALSTM-RMS-validation-best.pkl"
dst_model_file = "../ucf101-experiment/ucf101-rgb-motion-convALSTM/rms-lr-0.001-drop-0.7/UCF101-VideoModel-RGB-Motion-convALSTM-RMS-validation-best.pkl"
save_model_file = "../ucf101-experiment/ucf101-rgb-motion-convALSTM/rms-lr-0.001-drop-0.7/UCF101-VideoModel-RGB-Motion-convALSTM-RMS-initialization.pkl"

log_path = "model_assemble_ucf101_rgb_motion_convALSTM.log"
sparnn.utils.quick_logging_config(log_path)

############################# load model
src1_model = VideoModel.load(src1_model_file)
src1_model.print_stat()

src2_model = VideoModel.load(src2_model_file)
src2_model.print_stat()

dst_model = VideoModel.load(dst_model_file)
dst_model.print_stat()


# In[3]:

############################# model assemble
#copy_layers = range(len(src_model.middle_layers) -1 -1)
#for ii in copy_layers:
#    if len(src_model.middle_layers[ii].param)>0 and len(dst_model.middle_layers[ii].param)>0:
#        for src_param, dst_param in zip(src_model.middle_layers[ii].param, dst_model.middle_layers[ii].param):
#            assert src_param.name == dst_param.name
#            assert src_param.get_value().shape == dst_param.get_value().shape
#            dst_param.set_value(src_param.get_value())    
dst_model.middle_layers[4].param, src1_model.middle_layers[0].param, src2_model.middle_layers[2].param
#print len(dst_model.middle_layers[4].param), len(src1_model.middle_layers[0].param), len(src2_model.middle_layers[2].param)
#print src2_model.middle_layers[3].param[0].name, dst_model.middle_layers[5].param[0].name


# In[4]:

############################## bottom layer
dst_model.middle_layers[4].param[12].set_value( src1_model.middle_layers[0].param[0].get_value() )
dst_model.middle_layers[4].param[13].set_value( src1_model.middle_layers[0].param[1].get_value() )
dst_model.middle_layers[4].param[14].set_value( src1_model.middle_layers[0].param[2].get_value() )

dst_model.middle_layers[4].param[16].set_value( src1_model.middle_layers[0].param[3].get_value() )
dst_model.middle_layers[4].param[17].set_value( src1_model.middle_layers[0].param[4].get_value() )
dst_model.middle_layers[4].param[18].set_value( src1_model.middle_layers[0].param[5].get_value() )

dst_model.middle_layers[4].param[20].set_value( src1_model.middle_layers[0].param[6].get_value() )
dst_model.middle_layers[4].param[21].set_value( src1_model.middle_layers[0].param[7].get_value() )
dst_model.middle_layers[4].param[22].set_value( src1_model.middle_layers[0].param[8].get_value() )

dst_model.middle_layers[4].param[24].set_value( src1_model.middle_layers[0].param[9].get_value() )
dst_model.middle_layers[4].param[25].set_value( src1_model.middle_layers[0].param[10].get_value() )
dst_model.middle_layers[4].param[26].set_value( src1_model.middle_layers[0].param[11].get_value() )

############################## top layer
dst_model.middle_layers[4].param[0].set_value( src2_model.middle_layers[2].param[0].get_value() )
dst_model.middle_layers[4].param[1].set_value( src2_model.middle_layers[2].param[1].get_value() )
dst_model.middle_layers[4].param[2].set_value( src2_model.middle_layers[2].param[2].get_value() )

dst_model.middle_layers[4].param[3].set_value( src2_model.middle_layers[2].param[3].get_value() )
dst_model.middle_layers[4].param[4].set_value( src2_model.middle_layers[2].param[4].get_value() )
dst_model.middle_layers[4].param[5].set_value( src2_model.middle_layers[2].param[5].get_value() )

dst_model.middle_layers[4].param[6].set_value( src2_model.middle_layers[2].param[6].get_value() )
dst_model.middle_layers[4].param[7].set_value( src2_model.middle_layers[2].param[7].get_value() )
dst_model.middle_layers[4].param[8].set_value( src2_model.middle_layers[2].param[8].get_value() )

dst_model.middle_layers[4].param[9].set_value( src2_model.middle_layers[2].param[9].get_value() )
dst_model.middle_layers[4].param[10].set_value( src2_model.middle_layers[2].param[10].get_value() )
dst_model.middle_layers[4].param[11].set_value( src2_model.middle_layers[2].param[11].get_value() )

############################## attention layer
dst_model.middle_layers[4].param[28].set_value( src2_model.middle_layers[2].param[12].get_value() )
dst_model.middle_layers[4].param[29].set_value( src2_model.middle_layers[2].param[13].get_value() )
dst_model.middle_layers[4].param[30].set_value( src2_model.middle_layers[2].param[14].get_value() )
dst_model.middle_layers[4].param[31].set_value( src2_model.middle_layers[2].param[15].get_value() )
dst_model.middle_layers[4].param[32].set_value( src2_model.middle_layers[2].param[16].get_value() )

############################## copy full layers
from_layers = [0, 1, 3, 5]
to_layers = [0, 1, 5, 7]
for ii, jj in zip(from_layers, to_layers):
    if len(src2_model.middle_layers[ii].param)>0 and len(dst_model.middle_layers[jj].param)>0:
        for src_param, dst_param in zip(src2_model.middle_layers[ii].param, dst_model.middle_layers[jj].param):
            #assert src_param.name == dst_param.name
            assert src_param.get_value().shape == dst_param.get_value().shape
            dst_param.set_value(src_param.get_value())


# In[5]:

VideoModel.save(dst_model, save_model_file)


# In[ ]:

