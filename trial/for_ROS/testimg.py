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
tc_g = vm.getcamerapose(img.copy())


# redtracker
tracker = RedTracker(img.copy(),showimage=1,initialize_with_hand=1)
tracker.track(img)
pos=tracker.getpos()

# see the mask
#mask,centers,flag,validnum = tracker.extractRed(img)
#cv2.imshow("Mask",mask)

depth = np.loadtxt('dimage.csv',delimiter=",")
cv2.waitKey(0)

import math
print(pos)
print(depth[math.floor(pos[1]):math.floor(pos[1])+2,math.floor(pos[0]):math.floor(pos[0])+2])


cv2.destroyAllWindows()

# 3D pose on camera coordinate
Z = depth[int(pos[1]),int(pos[0])]
X = Z * pos[0]
Y = Z * pos[1]

tc = vm.tvecs.reshape(3,1)
Rc,_ = cv2.Rodrigues(vm.rvecs)

to_c = np.dot(np.linalg.inv(K),np.array([X,Y,Z]).reshape(3,1))

print("Stereo Based Method")
print(tc_g+np.dot(Rc.T,to_c))

# proposed method 1

print("Proposed Method")
print(vm.getobjpose_1(pos,0.12))