#!/usr/bin/env python
# -*- coding: utf-8 -*
# assuming python2 for ROS

import cv2
import numpy as np
import sys
import math

sys.path.append("../")

from vmarker import *


def extractRed(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_min = np.array([0,127,0])
    hsv_max = np.array([2,255,255])
    mask1  = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    hsv_min = np.array([150,150,0])
    hsv_max = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)
    
    # RGB search
    bgr_min = np.array([0,0,120])
    bgr_max = np.array([50,50,255])
    mask3 = cv2.inRange(img,bgr_min, bgr_max)

    Mmt = cv2.moments(mask2+mask3)
    if Mmt["m00"] != 0:
        cx = Mmt['m10']/Mmt['m00']
        cy = Mmt['m01']/Mmt['m00']
        flag = True
    else:
        cx,cy = 0,0
        flag = False
    #print([cx,cy])
    return mask2,[cx,cy],flag

if __name__=='__main__':
    import sys
    args = sys.argv
    if len(args)>1:
        filename = args[1]
    else:
        filename = "output.avi"
    cap = cv2.VideoCapture(filename)
    # load camera matrix and distort matrix
    K = np.loadtxt("../calib_usb/K.csv",delimiter=",")
    dist_coef = np.loadtxt('../calib_usb/d.csv',delimiter=",")
    vm = vmarker(markernum=5,K=K,dist=dist_coef,markerpos_file="roomA_ground_orig.csv")
    
    try:
        while ~cap.isOpened():
            ok,frame = cap.read()
            #nframe = cv2.undistort(frame, K, dist_coef)
            mask,cpts,flag = extractRed(frame)
            if vm.PNPsolved*flag:
                objxy = vm.getobjpose_1(cpts,0.13)
                print([objxy[0],objxy[1]])
            else:
                tv = vm.getcamerapose(frame)
                cv2.imwrite('extraction.png',frame)
            vm.showmarker(frame)
            cv2.imshow("mask",mask)
            cv2.waitKey(1)


    except KeyboardInterrupt:
        print("Finish Program!")
        exit(0)
