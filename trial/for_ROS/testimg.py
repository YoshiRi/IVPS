#!/usr/bin/env python
# -*- coding: utf-8 -*
# assuming python2.7

import cv2
import numpy as np
import sys

sys.path.append('../../')
from RedTracker import *
from vmarker import *

img = cv2.imread("limage.png")
#img_fix = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)# true color?

# do not work well... noooooooooo
K = np.loadtxt("vm_K.csv",delimiter=",").reshape(3,3)
dist_coef = np.loadtxt('vm_D.csv',delimiter=",")
vm = vmarker(K=K,dist=dist_coef,markerpos_file="roomA_ground_orig.csv")
vm.getcamerapose(img.copy())


# redtracker
tracker = RedTracker(img.copy(),showimage=1,initialize_with_hand=1)
tracker.track(img)
pos=tracker.getpos()

# see the mask
mask,centers,flag,validnum = tracker.extractRed(img)

cv2.imshow("Mask",mask)
depth = np.loadtxt('dimage.csv',delimiter=",")

cv2.waitKey(0)

print(pos)
print(depth[int(pos[1]),int(pos[0])])