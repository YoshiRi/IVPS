#!/usr/bin/env python
# -*- coding: utf-8 -*
# assuming python2 for ROS

import cv2
import numpy as np
import sys
import math

sys.path.append("../")

from vmarker import *


if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    # load camera matrix and distort matrix
    K = np.loadtxt("../calib_usb/K.csv",delimiter=",")
    dist_coef = np.loadtxt('../calib_usb/d.csv',delimiter=",")
    vm = vmarker(markernum=4,K=K,dist=dist_coef,markerpos_file="roomA_r.csv")
    try:
        while ~cap.isOpened():
            ok,frame = cap.read()
            nframe = cv2.undistort(frame, K, dist_coef)
            #detect = detect_marker(frame)
            #cv2.imshow("detected",detect)
            #cv2.waitKey(1)
            tv = vm.getcamerapose(frame)
            print(tv)

    except KeyboardInterrupt:
        print("Finish Program!")
        exit(0)