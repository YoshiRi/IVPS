#!/usr/bin/env python
# -*- coding: utf-8 -*
# assuming python3 and OpenCV 3.X

"""Process Video and Get Camera Pose

usage:
    test.py [-h] [--display_image] [--init_hand] [--filename <fn>] [--sizeofwindow <s>] [--objectheight <hei>] [--camerainformation <camerafile>]

options:
    -h, --help  show this help message and exit
    -d, --display_image    display processed image
    -i, --init_hand    initialize the tracking region with hand
    -f, --filename <fn>    filename for video
    -s, --sizeofwindow <s>    windowsize for tracking
    -o, --objectheight <dep> tracking vehicle height
    -c, --camerainformation <camerafile> camera position file
"""


import cv2
import numpy as np
import sys
import math
from docopt import docopt
import matplotlib.pyplot as plt


sys.path.append("../")

from vmarker import *
from RedTracker import *


if __name__=='__main__':
    args = docopt(__doc__)

    if args["--filename"]:
        fname = args["--filename"]
    else:
        fname = 0

    if args["--objectheight"]:
        oheight = args["--objectheight"]
    else:
        oheight = 0.13 # only true for test environment




    cap = cv2.VideoCapture(fname)
    # load camera matrix and distort matrix
    K = np.loadtxt("../calib_usb/K.csv",delimiter=",")
    dist_coef = np.loadtxt('../calib_usb/d.csv',delimiter=",")
    vm = vmarker(markernum=5,K=K,dist=dist_coef,markerpos_file="roomA_ground_orig.csv")

    # 1. extract camera pose
    if args["--camerainformation"]: # has camera yaml file
        try:
            vm.load_camerapose_yml(args["--camerainformation"])
        except:
            print("unable to open file:"+args["--camerainformation"])
    else:
        while ~cap.isOpened():
            ok,frame = cap.read()
            if not ok:
                break
            tv=vm.getcamerapose(frame)
            if vm.hasCameraPose:
                break
            vm.showmarker(frame)
            cv2.waitKey(1)
    cv2.destroyAllWindows()



    # tracking and get pose

    # init tracker
    ok,frame = cap.read()

    if args["--sizeofwindow"]:
        tracker = RedTracker(frame, showimage=args["--display_image"], initialize_with_hand=args["--init_hand"], bboxsize=args["--sizeofwindow"])
    else:
        tracker = RedTracker(frame, showimage=args["--display_image"], initialize_with_hand=args["--init_hand"])

    pos2d = []
    pos3d = []
    # start video stream
    try:
        lines = []
        while ~cap.isOpened():
            ok,frame = cap.read()
            if not ok:
                break
            tracker.track(frame)
            objxy = vm.getobjpose_1(tracker.getpos(), oheight)
            ## print(str(tracker.getpos())+str(objxy))
            
            pos2d.append(tracker.pos)
            pos3d.append(objxy)
            

    except KeyboardInterrupt:
        print("Finish Program!")

    np.savetxt("pos2d.txt",np.array(pos2d))
    np.savetxt("pos3d.txt",np.array(pos3d))
    cap.release()
    cv2.destroyAllWindows()

