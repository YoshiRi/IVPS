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

# calculate bird view homography
import math
theta_below = np.array([0, math.pi, 0])
Rc_, _ =  cv2.Rodrigues(theta_below)
Rc,_ = cv2.Rodrigues(vm.rvecs)


Homo = np.linalg.multi_dot([K,Rc_.T,Rc.T,np.linalg.inv(K)])

print(Homo)

nHomo = Homo/Homo[2,2]

print(nHomo)

minx = -20000
miny = 0

nHomo[0,2] -= minx
nHomo[1,2] -= miny

print(nHomo)

out=cv2.warpPerspective(img,nHomo,(2000,2000))

cv2.imshow("warped",out)
cv2.waitKey(0)

cv2.imwrite("birdview.png",out)