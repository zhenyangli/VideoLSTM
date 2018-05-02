#!/usr/bin/python

import argparse
import sys, os
import errno
import subprocess
from os import listdir
from os.path import isfile, join
import json
import glob
from PIL import Image

caffelib = '/home/zhenyang/local/softs/caffe'

if caffelib:
    caffepath = caffelib + '/python'
    sys.path.append(caffepath)

import caffe
from extract_features_rgbcnn_fc import batch_predict


def caffe_init(use_gpu, model_def_file, model_file, gpu_id):
    """
    Initilize pycaffe wrapper
    """

    if use_gpu:
        print 'Using GPU Mode'
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    else:
        print 'Using CPU Mode'
        caffe.set_mode_cpu()

    # By default use imagenet_deploy
    # model_def_file = 'models/UCF_CNN_M_2048_deploy.prototxt'
    # By default use caffe reference model
    # model_file = 'models/1_vgg_m_fine_tuning_rgb_iter_20000.caffemodel'
    if os.path.isfile(model_file):
        # NOTE: you'll have to get the pre-trained ILSVRC network
        print 'You need a network model file'

    if os.path.isfile(model_def_file):
        # NOTE: you'll have to get network definition
        print 'You need the network prototxt definition'

    # run with phase test (so that dropout isn't applied)
    net = caffe.Net(model_def_file, model_file, caffe.TEST)
    #caffe.set_phase_test()
    print 'Done with init, Done with set_phase_test'

    return net


def getImageFeatures(net, inputfile, outputfile):
    if not os.path.exists(outputfile+'.h5'):
        print '(3/3) getImageFeatures: ' + inputfile
        batch_predict(inputfile, outputfile, net)
    else:
        print '(3/3) getImageFeatures: ' + inputfile + ' Exist: '+ outputfile


def addToList(net, inputdir, framefreq):
    print '(2/3) addToList: ' + inputdir

    _inputdir = inputdir.replace('[', '[[]')
    frames = glob.glob(join(_inputdir, '*image_*.jpg'))
    duration = len(frames) - 1 # why one less??
    assert duration>0
    counter = framefreq

    open(inputdir + '/tasks.txt', 'w').close()
    for i in range(duration):
        frame = join(inputdir, 'image_{0:04d}.jpg'.format(i+1))
        if frame.endswith('.jpg') or frame.endswith('.png'):
            if counter >= framefreq:
                #print frame
                with open(inputdir + '/tasks.txt', 'a') as textfile:
                    textfile.write(frame + '\n')
                counter = 0
            counter += 1

    inputfile = inputdir + '/tasks.txt'
    outputfile = join(os.path.dirname(os.path.dirname(inputdir)), 'rgb_vgg16_fc7', os.path.basename(inputdir))
    getImageFeatures(net,inputfile,outputfile)


def extractVideo(net, inputdir, outputdir, framefreq):
    #if not os.path.exists(outputdir):
    if 0:
        os.makedirs(outputdir)
        print('(1/3) extractVideo: ' + inputdir + ' To: ' + outputdir)

        # get framerate
        command = 'ffmpeg -i '+inputdir+' 2>&1 | sed -n "s/.*, \(.*\) fp.*/\\1/p"'
        framerate = '30'
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
        while proc.poll() is None:
            line = proc.stdout.readline()
            if len(line) > 1: framerate = line.rstrip('\n')

        print 'framerate: '+framerate

        # get video
        command = 'ffmpeg -i '+inputdir+' -framerate '+str(framerate)+' -y -f image2 '+outputdir+'/frame-%06d.jpg'
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
        while proc.poll() is None:
            line = proc.stdout.readline()
            #print(line + '\n')

        #print('extracting audio: '+inputdir+' To: '+outputdir)

        # get audio
        #command = 'ffmpeg -i '+inputdir+' -y -map 0:1 -vn -acodec copy '+outputdir+'/output.aac'
        #proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
        #while proc.poll() is None:
            #line = proc.stdout.readline()
            ##print(line + '\n')
    else:
        print('(1/3) extractVideo: ' + inputdir + ' Exist: ' + outputdir)

    addToList(net,outputdir,framefreq)


# python extract_rgbcnn.py -s 1 -t 1600 --gpu_id 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FeatureExtractior')
    parser.add_argument('-d', '--dataset', dest='dataset', help='Specify dataset to process.', type=str, required=False)
    parser.add_argument('-s', '--startvid', dest='startvid', help='Specify video id start to process.', type=int, required=False)
    parser.add_argument('-t', '--tovid', dest='tovid', help='Specify video id until to process.', type=int, required=False)
    parser.add_argument('-f', '--frequency', dest='frequency', help='Specify frame frequency to extract.', type=int, required=False)
    parser.add_argument('--model_def', dest='model_def', type=str, default='bvlc_googlenet_deploy_features.prototxt')
    parser.add_argument('--model', dest='model', type=str, default='bvlc_googlenet.caffemodel')
    parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0)
    args = parser.parse_args()

    if args.dataset is None:
        print 'Not specify dataset, using UCF101 dataset by default...'
        args.dataset = '/home/zhenyang/Workspace/data/UCF101/list_UCF101.txt'
    if args.frequency is None:
        args.frequency = 1

    print '***************************************'
    print '********** EXTRACT FEATURES ***********'
    print '***************************************'
    print 'Dataset: %s' % (args.dataset, )
    print 'Frame frequency: %d' % (args.frequency, )

    data_dir = os.path.dirname(args.dataset)
    
    filenames = []
    with open(args.dataset) as fp:
        for line in fp:
            splits = line.strip().split(' ')
            filenames.append(splits[0])

    Nf = len(filenames)
    startvid = 0
    toid = Nf
    if args.startvid is not None and args.tovid is not None:
        startvid = max([args.startvid-1, startvid])
        toid = min([args.tovid, toid])

    # initialize caffe network
    net = caffe_init(1, args.model_def, args.model, args.gpu_id)

    for i in range(startvid, toid):
        
        filename = filenames[i]
        filename_ = os.path.splitext(os.path.basename(filename))[0]
        
        #filename_ = filenames[i]
        #filename = filename_ + '.avi'

        print 'Processing (%d/%d): %s' % (i+1,Nf,filename, )

        videofile = join(data_dir, 'videos', filename)
        outputfile = join(data_dir, 'features', 'flow_tvl1_gpu', filename_)

        extractVideo(net,videofile,outputfile,args.frequency)


    print '*********** PROCESSED ALL *************'
