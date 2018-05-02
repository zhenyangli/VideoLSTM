import sys
import os.path

import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import cv2
import h5py
import cPickle as pickle

caffelib = '/home/zhenyang/local/softs/caffe'

if caffelib:
    caffepath = caffelib + '/python'
    sys.path.append(caffepath)

import caffe

def predict(in_data, net):
    """
    Get the features for a batch of data using network

    Inputs:
    in_data: data batch
    """

    out = net.forward(**{net.inputs[0]: in_data})
    #features = out[net.outputs[0]].squeeze(axis=(2,3))
    features = out[net.outputs[0]]
    return features


def batch_predict(listfile, outputfile, net):
    """
    Get the features for all images from a file list using a network

    Inputs:
    listfile: a file containing a list of names of image files

    Returns:
    an array of feature vectors for the images in that file
    """

    filenames_x = []
    filenames_y = []
    base_dir = os.path.dirname(listfile)
    with open(listfile) as fp:
        for line in fp:
            splits = line.strip().split(' ')
            filenames_x.append(splits[0])
            filenames_y.append(splits[1])

    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    N, F, H_conv, W_conv = net.blobs[net.outputs[0]].data.shape
    Nf = len(filenames_x)
    #Hi, Wi, _ = imread(filenames[0]).shape
    #allftrs = np.zeros((Nf, F*H_conv*W_conv), dtype=np.float32)
    allftrs = np.zeros((Nf, F, H_conv, W_conv), dtype=np.float32)
    for i in range(0, Nf, N):
        in_data = np.zeros((N, C, H, W), dtype=np.float32)

        batch_range = range(i, min(i+N, Nf))
        batch_filenames_x = [filenames_x[j] for j in batch_range]
        batch_filenames_y = [filenames_y[j] for j in batch_range]
        Nb = len(batch_range)

        batch_images = np.zeros((Nb, 2, H, W))
        for j,fname_x,fname_y in zip(range(Nb), batch_filenames_x, batch_filenames_y):
            #im = imread(fname)
            im_x = cv2.imread(fname_x, cv2.IMREAD_GRAYSCALE)
            im_y = cv2.imread(fname_y, cv2.IMREAD_GRAYSCALE)
            # RGB -> BGR, single channel and already BGR

            # first, resize (scipy.misc.imresize only works with uint8)
            # We turn off Matlab's antialiasing to better match OpenCV's bilinear 
            # interpolation that is used in Caffe's WindowDataLayer.
            # im = imresize(im, (H, W), 'bilinear')
            im_x = cv2.resize(im_x, (W, H), interpolation=cv2.INTER_LINEAR)
            im_y = cv2.resize(im_y, (W, H), interpolation=cv2.INTER_LINEAR)
            # then, mean subtraction
            batch_images[j,0,:,:] = im_x - 128.0
            batch_images[j,1,:,:] = im_y - 128.0

        # insert into correct place
        in_data[0:len(batch_range), :, :, :] = batch_images

        # predict features
        ftrs = predict(in_data, net)

        # transpose (N,C,H,W) -> (N,H,W,C)
        # ftrs = np.transpose(ftrs, (0,2,3,1))
        # reshape into single feature vector with length F*H_conv*W_conv
        # ftrs = ftrs.reshape((N, F*H_conv*W_conv))

        #for j in range(len(batch_range)):
        #    allftrs[i+j,:] = ftrs[j,:]
        for j in range(len(batch_range)):
            allftrs[i+j,:,:,:] = ftrs[j,:,:,:]

        print 'Done %d/%d files' % (i+len(batch_range), len(filenames_x))

    if outputfile:
        # store the features in a pickle file
        #with open(outputfile, 'w') as fp:
        #    pickle.dump(allftrs, fp)

        # store the features in a hdf5 file
        with h5py.File(outputfile+'.h5', "w") as fp:
            dset = fp.create_dataset("features", data=allftrs)

    return allftrs

