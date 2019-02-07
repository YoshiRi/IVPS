#!/usr/bin/env python
# -*- coding: utf-8 -*
# assuming python3 and OpenCV 3.X

"""Process Video and Get Camera Pose

usage:
    test.py [-h] [--display_image] [--init_hand] [--filename <fn>] [--sizeofwindow <s>] [--objectheight <hei>] [--camerainformation <camerafile>] [--allow3ptsestimate]

options:
    -h, --help  show this help message and exit
    -d, --display_image    display processed image
    -i, --init_hand    initialize the tracking region with hand
    -f, --filename <fn>    filename for video
    -s, --sizeofwindow <s>    windowsize for tracking
    -o, --objectheight <dep> tracking vehicle height
    -c, --camerainformation <camerafile> camera position file
    -a, --allow3ptsestimate     allow use only 3 points to estimate camera pose
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
    vm = vmarker(markernum=5,K=K,dist=dist_coef,markerpos_file="data/roomA_ground_orig.csv",showimage=args["--display_image"])

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
            tv=vm.getcamerapose(frame,allow3pts=args["--allow3ptsestimate"],rvec_init=[1.73052624,1.26932939,-0.6861009],tvec_init=[0.26549237,0.28860147,3.46103553])
            if vm.hasCameraPose:
                print('Camera Poses R,T is:'+str(vm.rvecs.T)+str(vm.tvecs.T))
                break
            if args["--display_image"]:
                vm.showmarker(frame)
                cv2.waitKey(1)
    if args["--display_image"]:
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

    if args["--display_image"]:
        plt.plot([i[0] for i in pos3d],[i[1] for i in pos3d])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.show()

        np.savetxt("pos2d.txt",np.array(pos2d))
        np.savetxt("pos3d.txt",np.array(pos3d))
    
    print('Average and std')
    print(np.mean(np.array(pos3d).reshape(-1,2),axis=0))
    print(np.std(np.array(pos3d).reshape(-1,2),axis=0))
    cap.release()
    cv2.destroyAllWindows()

