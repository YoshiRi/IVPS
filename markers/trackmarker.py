#!/usr/bin/env python
# -*- coding: utf-8 -*
# assuming python2 for ROS

import cv2
import numpy as np
import sys
import math

sys.path.append("../")

from vmarker import *


def detect_marker1(frame):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)
    dimage = aruco.drawDetectedMarkers(frame.copy(),corners)
    return dimage

def detect_marker2(frame):# detect smallmarkers
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary,parameters=param)
    dimage = aruco.drawDetectedMarkers(frame.copy(),corners)
    dfimage = aruco.drawDetectedMarkers(frame.copy(),rejectedImgPoints)
    
    return dimage,dfimage


param= cv2.aruco.DetectorParameters_create()
param.minDistanceToBorder = 1

if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    args = sys.argv
    duration = 10000 # 10000 sample = 333s = 5.5m
    if len(args) > 1:
        duration = int(args[1])
    # Define the codec and create VideoWriter object
    

    try:
        while ~cap.isOpened() and (duration > 0 ):
            ret,frame = cap.read()
            duration = duration - 1
            if ret==True:

                dframe,dfframe = detect_marker2(frame)
                cv2.imshow('frame',dframe)
                cv2.imshow('frame candidate',dfframe)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("Finish Program!")
        cap.release()
        exit(0)
