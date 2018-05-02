### Requirements

    Latest Python2 (python2.7.*)
    numpy + scipy
    Theano
    HDF5

### How to Install
    
    Official Installation Guide For Related Packages
        
        Numpy & Scipy:
            http://docs.scipy.org/doc/numpy/user/install.html
        Theano:
            http://deeplearning.net/software/theano/install.html
        HDF5:
            https://hdfgroup.org/HDF5/

### How to Config Theano
Theano is the backbone of this project. To configure theano, view [theano-config](http://deeplearning.net/software/theano/library/config.html) for more detailed help. You need to write the configuration to ~/.theanorc. The followingTheano configuration is recommended

For CPU users:

    [global]
    floatX = float32
    device = cpu
    mode = FAST_RUN
    warn_float64 = warn

For GPU users(here the device can be any other GPU):

    [global]
    floatX = float32
    device = gpu0
    mode = FAST_RUN
    warn_float64 = warn


=======================================================================

### Library

4 main components:

`iterator`: data handler

`layer`: network layers, to construct a network, you have to have 3 kinds of layers
```
interface layer: declare input, mask, output
middle layer: construct the main network layers (a list of layers)
cost layer: construct the network cost
```
`model`: network model

`optimizer`: optimizer to optimize the model


### Data Format
The `data_file` is an folder path with a list of hdf5 files for videos:
```
v_ApplyEyeMakeup_g08_c01.h5
v_ApplyEyeMakeup_g08_c02.h5
v_ApplyEyeMakeup_g08_c03.h5
v_ApplyEyeMakeup_g08_c04.h5
v_ApplyEyeMakeup_g08_c05.h5
```
Each hdf5 file stores all the frame features for this video row by row, i.e., a matrix with size (#frames, #featureDim)

The `train_framenum.txt` file contains number of frames for each video:
```
89
123
22
136
```

The `train_filenames.txt` file contains the video filenames relative to the root video directory:
```
v_ApplyEyeMakeup_g08_c01
v_ApplyEyeMakeup_g08_c02
v_ApplyEyeMakeup_g08_c03
v_ApplyEyeMakeup_g08_c04
v_ApplyEyeMakeup_g08_c05
```

The `train_labels.txt`file for single-label datasets looks like
```
0
7
43
```
and for multi-label datasets:
```
0,0,0,0,0,0,0,1,0,0,0,0
0,0,0,0,0,0,0,1,0,0,0,0
0,0,0,0,0,0,1,1,0,0,0,0
0,0,0,0,0,0,0,0,0,0,0,1
```
The same format is required for the validation and test files too.


=======================================================================

### Network Architectures

LSTM: LSTM

ALSTM: Attention LSTM

convLSTM: Convolutional LSTM

convALSTM: Convolutional Attention LSTM

motion ALSTM: Attention LSTM with motion-based attention

motion convALSTM: Convolutional Attention LSTM with motion-based attention



Example to run the scripts:

```
THEANO_FLAGS='floatX=float32,device=gpu0,mode=FAST_RUN,nvcc.fastmath=True' python evaluate_ucf101_rgb_LSTM.py
THEANO_FLAGS='floatX=float32,device=gpu1,mode=FAST_RUN,nvcc.fastmath=True' python evaluate_ucf101_flow_ALSTM.py
```


=======================================================================

### Feature Extraction


We use `extract_rgbcnn.py` and `extract_flowcnn.py` scripts to extract feature maps (e.g. pool5 features) for rgb and flow input.
While `extract_rgbcnn_fc.py` and `extract_rgbcnn_fc.py` are used to extract fc features.

Caffe network definition files and models:

`rgb`

prototxt: ucf101_action_rgb_vgg_16_deploy_features_fc7.prototxt, ucf101_action_rgb_vgg_16_deploy_features_pool5.prototxt

model: ucf101_action_rgb_vgg_16_split1.caffemodel


`flow (single flow)`

prototxt: ucf101_action_singleflow_vgg_16_deploy_features_fc7.prototxt, ucf101_action_singleflow_vgg_16_deploy_features_pool5.prototxt

model: ucf101_action_singleflow_vgg_16_split1.caffemodel


Example to run the scripts:
```
python extract_rgbcnn.py --model_def ucf101_action_rgb_vgg_16_deploy_features_pool5.prototxt --model ucf101_action_rgb_vgg_16_split1.caffemodel --gpu_id 0
```
