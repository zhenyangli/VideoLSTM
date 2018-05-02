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

    filenames = []
    base_dir = os.path.dirname(listfile)
    with open(listfile) as fp:
        for line in fp:
            filename = os.path.join(base_dir, line.strip().split()[0])
            filenames.append(filename)

    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    F = net.blobs[net.outputs[0]].data.shape[1]
    Nf = len(filenames)
    #Hi, Wi, _ = imread(filenames[0]).shape
    allftrs = np.zeros((Nf, F), dtype=np.float32)
    for i in range(0, Nf, N):
        in_data = np.zeros((N, C, H, W), dtype=np.float32)

        batch_range = range(i, min(i+N, Nf))
        batch_filenames = [filenames[j] for j in batch_range]
        Nb = len(batch_range)

        batch_images = np.zeros((Nb, 3, H, W))
        for j,fname in enumerate(batch_filenames):
            #im = imread(fname)
            im = caffe.io.load_image(fname)*255.0

            if len(im.shape) == 2:
                im = np.tile(im[:,:,np.newaxis], (1,1,3))
            # RGB -> BGR
            im = im[:,:,(2,1,0)]
            # first, resize (scipy.misc.imresize only works with uint8)
            # We turn off Matlab's antialiasing to better match OpenCV's bilinear 
            # interpolation that is used in Caffe's WindowDataLayer.
            # im = imresize(im, (H, W), 'bilinear')
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
            # then, mean subtraction
            im = im - np.array([104., 117., 123.])
            # get channel in correct dimension
            im = np.transpose(im, (2,0,1))
            batch_images[j,:,:,:] = im

        # insert into correct place
        in_data[0:len(batch_range), :, :, :] = batch_images

        # predict features
        ftrs = predict(in_data, net)

        for j in range(len(batch_range)):
            allftrs[i+j,:] = ftrs[j,:]

        print 'Done %d/%d files' % (i+len(batch_range), len(filenames))

    if outputfile:
        # store the features in a pickle file
        #with open(outputfile, 'w') as fp:
        #    pickle.dump(allftrs, fp)

        # store the features in a hdf5 file
        with h5py.File(outputfile+'.h5', "w") as fp:
            dset = fp.create_dataset("features", data=allftrs)

    return allftrs

