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
    args = sys.argv
    duration = 10000 # 10000 sample = 333s = 5.5m
    if len(args) > 1:
        duration = int(args[1])
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    try:
        while ~cap.isOpened() and (duration > 0 ):
            ret,frame = cap.read()
            duration = duration - 1
            if ret==True:

                # write the flipped frame
                out.write(frame)            

                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("Finish Program!")
        cap.release()
        out.release()
        exit(0)
