#!/usr/bin/python

import argparse
import os
import errno
import subprocess
import shutil
from os import listdir
from os.path import isfile, join
import json
import re
import glob


flowgpu_bin = './denseFlow_gpu'


def getVideoFlowFeatures(inputfile,outputfile,gpu_id):
    print '(1/1) getVideoFlowFeatures: ' + inputfile

    #'./denseFlow -f ',vid_name,' -x test/flow_x -y test/flow_y -b 20'
    #'./denseFlow_gpu -f ',vid_name,' -x test/flow_x -y test/flow_y -b 20 -t 1 -d 3'

    inputfile = re.escape(inputfile)
    outputfile = re.escape(outputfile)
    command = '%s -f %s -x %s/flow_x -y %s/flow_y -z %s/flow -i %s/image -b 20 -t 1 -d %d -s 1' % (flowgpu_bin,inputfile,outputfile,outputfile,outputfile,outputfile,gpu_id,)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
    while proc.poll() is None:
        line = proc.stdout.readline()
        print(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FeatureExtractior')    
    parser.add_argument('-d', '--dataset', dest='dataset', help='Specify dataset to process.', type=str, required=False)
    parser.add_argument('-s', '--startvid', dest='startvid', help='Specify video id start to process.', type=int, required=False)
    parser.add_argument('-t', '--tovid', dest='tovid', help='Specify video id until to process.', type=int, required=False)
    parser.add_argument('-g', '--gpu_id', dest='gpu_id', help='Specify gpu device id to process.', type=int, required=False)
    args = parser.parse_args()

    if args.dataset is None:
        print 'Not specify dataset, using UCF101 by default...'
        args.dataset = '/home/zhenyang/Workspace/data/UCF101/list_UCF101.txt'

    print '***************************************'
    print '********** EXTRACT FEATURES ***********'
    print '***************************************'
    print 'Dataset: %s' % (args.dataset, )

    base_dir = os.path.dirname(args.dataset)

    filenames = []
    with open(args.dataset) as fp:
        for line in fp:
            filenames.append(line.strip())

    Nf = len(filenames)
    startvid = 0
    tovid = Nf
    if args.startvid is not None and args.tovid is not None:
        startvid = max([args.startvid-1, startvid])
        tovid = min([args.tovid, tovid])

    for i in range(startvid, tovid):

        filename = filenames[i]
        filename_ = os.path.splitext(os.path.basename(filename))[0]
        
        #filename_ = filenames[i]
        #filename = filename_ + '.avi'

        print 'Processing (%d/%d): %s' % (i+1,Nf,filename, )

        inputfile = os.path.join(base_dir, 'videos', filename)
        outputfile = os.path.join(base_dir, 'features', 'flow_tvl1_gpu', filename_)

        frames = glob.glob(join(outputfile, 'image_*.jpg'))
        duration = len(frames)

        if not os.path.exists(outputfile):
            os.makedirs(outputfile)
            getVideoFlowFeatures(inputfile,outputfile,args.gpu_id)

        if duration == 0:
            getVideoFlowFeatures(inputfile,outputfile,args.gpu_id)

    print '********* PROCESSED ALL ************'

