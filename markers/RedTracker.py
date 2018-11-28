#!/usr/bin/env python
# -*- coding: utf-8 -*
# assuming python2 for ROS

import cv2
import numpy as np


class RedZoneTracker:
    def __init__(self,frame,initialize=1,tracker='KCF'):
        # if init with hand:
        if initialize:
            

        self.get_tracker

    def get_tracker(self,name):
        """
        Choose tracker from key word
        """
        self.boxtracker = {
            'Boosting': cv2.TrackerBoosting_create(),
            'MIL': cv2.TrackerMIL_create(),
            'KCF' : cv2.TrackerKCF_create(),
            'TLD' : cv2.TrackerTLD_create(),
            'MedianFlow' : cv2.TrackerMedianFlow_create()
        }.get(name, 0)        

def extractROI(frame,roi):
    return frame[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]

def drawrect(frame,bbox,color=(0,255,0)):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, color, 2, 1)



def find_rect_of_target_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    mask = np.zeros(h.shape, dtype=np.uint8)
    mask[((h < 15) | (h > 200)) & (s > 128)] = 255
    # Get boundary
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    
    for contour in contours:
        approx = cv2.convexHull(contour)
        rect = cv2.boundingRect(approx)
        rects.append(np.array(rect))
    return max(rects, key=(lambda x: x[2] * x[3])) #return maximum rectangle


def extractRed(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    mask = np.zeros(h.shape, dtype=np.uint8)
    mask[((h < 10) | (h > 200)) & (s > 128)] = 255

    # get RED size
    validnum = sum(mask.reshape(-1))/255
    hei,wid = image.shape

    Mmt = cv2.moments(mask)
    if Mmt["m00"] != 0:
        cx = Mmt['m10']/Mmt['m00']
        cy = Mmt['m01']/Mmt['m00']
        flag = True
    else:
        cx,cy = 0,0
        flag = False
    #print([cx,cy])
    return mask,[cx,cy],flag,validnum