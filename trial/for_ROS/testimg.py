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
showimg = img.copy()
tc_g = vm.getcamerapose(showimg)

'''
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
print(vm.getobjpose_1(pos,0.085))

'''

'''
[[ 2.63469052  2.63568878]
 [ 2.63287401  2.6329    ]]
Stereo Based Method
[[ 0.05990568]
 [-0.02124139]
 [ 0.13482995]]
Proposed Method
[0.00079239195965262466, 0.00085919515168742368]
'''




rect = cv2.selectROI(img, False)
vm.showmarker(showimg)
crop_axis = tracker.extractROI(showimg,rect)

crop=tracker.extractROI(img,rect)
dcrop=tracker.extractROI(depth,rect)
dcrop2 = dcrop/np.amax(dcrop)*255.0
dcrop2 = cv2.normalize(dcrop, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
cv2.imshow("crop",crop)
cv2.imshow("dcrop",dcrop2)
cv2.waitKey(0)

cv2.imwrite("figures/crop_axis.png",crop_axis)
cv2.imwrite("figures/crop.png",crop)
cv2.imwrite("figures/crop_depth.png",dcrop2)