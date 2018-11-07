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

    # 赤色のHSVの値域2
    hsv_min = np.array([150,127,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)
    
    Mmt = cv2.moments(mask2)
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
    cap = cv2.VideoCapture(0)
    # load camera matrix and distort matrix
    K = np.loadtxt("../calib_usb/K.csv",delimiter=",")
    dist_coef = np.loadtxt('../calib_usb/d.csv',delimiter=",")
    vm = vmarker(markernum=5,K=K,dist=dist_coef,markerpos_file="roomA.csv")
    try:
        while ~cap.isOpened():
            ok,frame = cap.read()
            #nframe = cv2.undistort(frame, K, dist_coef)
            mask,cpts,flag = extractRed(frame)
            cv2.imshow("mask",mask)
            if vm.PNPsolved*flag:
                objxy = vm.getobjpose_1(cpts,-0.13)
                print([objxy[0] -1.088,objxy[1] -1.412])
            else:
                tv = vm.getcamerapose(frame)
                cv2.imwrite('extraction.png',frame)
            cv2.waitKey(1)
            

    except KeyboardInterrupt:
        print("Finish Program!")
        exit(0)