#!/usr/bin/env python
# -*- coding: utf-8 -*
# assuming python2.7

import cv2
import numpy as np
import sys

sys.path.append('../../')
from RedTracker import *
from vmarker import *

img = cv2.imread("testimg2.png")
img_fix = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)# true color?

# do not work well... noooooooooo
K = np.loadtxt("K.csv",delimiter=",")
dist_coef = np.loadtxt('d.csv',delimiter=",")
vm = vmarker(K=K,dist=dist_coef,markerpos_file="roomA_ground_orig.csv")

# redtracker
tracker = RedTracker(img_fix.copy(),showimage=1,initialize_with_hand=1)
