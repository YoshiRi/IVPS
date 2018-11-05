#!/usr/bin/env python
# -*- coding: utf-8 -*
# assuming python2 for ROS

import cv2
import numpy as np
import math

def create_marker(mnum,imsize=200):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    markers = []
    for i in range(mnum):
        marker = aruco.drawMarker(dictionary, i, imsize)
        markers.append(marker)
        cv2.imwrite('markers/marker'+str(i)+'.png', marker)

def detect_marker(frame):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)
    dimage = aruco.drawDetectedMarkers(frame,corners)
    return dimage

def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

class vmarker:
    def __init__(self,markernum=5,K=[],dist=[],markerpos_file="mpos1.csv"):
        aruco = cv2.aruco
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        #self.startvideo()
        self.mnum = markernum
        self.setmarker(markerpos_file)
        self.K = K
        self.dist = dist
        self.tvecs = []
        self.rvecs = []
        self.R = []
    
    def setmarker(self,fname):
        #self.objp = np.zeros((markernum,3), np.float32)
        self.objp = np.loadtxt(fname,delimiter=",")
        self.mnum , _ = self.objp.shape
        print(self.mnum)
    
    def startvideo(self,vnum=0):
        self.cap = cv2.VideoCapture(0)
    
    def showmarker(self,frame):
        aruco = cv2.aruco
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, self.dictionary)
        detect = aruco.drawDetectedMarkers(frame,corners)
        cv2.imshow("detected",detect)
        cv2.waitKey(1)
    
    def getcamerapose(self,frame):
        aruco = cv2.aruco
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, self.dictionary)

        if len(corners) == self.mnum:
            # sort based on IDs and use center value
            centercorners = []
            for _,corner in sorted(zip(ids,corners)): #corner=[x11,y11]...
                centercorners.append(np.average(corner,1))
                
            self.ccorners = np.array(centercorners).reshape(self.mnum,1,2)
            #print(self.ccorners)

            # Find the rotation and translation vectors.
            _, self.rvecs, self.tvecs, inliers = cv2.solvePnPRansac(self.objp, self.ccorners, self.K, self.dist)
            self.drawaxis(aruco.drawDetectedMarkers(frame,corners,ids)) # draw origin
            self.R = eulerAnglesToRotationMatrix(self.rvecs)
            return -np.dot(self.R.T,self.tvecs)

        else:
            cv2.imshow('projected',aruco.drawDetectedMarkers(frame,corners,ids))
            cv2.waitKey(1)
            return []

    def draw(self,img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img

    def drawaxis(self,frame):#30cm cube
        self.axis = np.float32([[0.3,0,0], [0,0.3,0], [0,0,0.3]]).reshape(-1,3)
        imgpts, jac = cv2.projectPoints(self.axis, self.rvecs, self.tvecs, self.K, self.dist)
        img = self.draw(frame,self.ccorners,imgpts)
        cv2.imshow('projected',img)
        cv2.waitKey(1)

if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    # load camera matrix and distort matrix
    K = np.loadtxt("calib_usb/K.csv",delimiter=",")
    dist_coef = np.loadtxt('calib_usb/d.csv',delimiter=",")
    vm = vmarker(K=K,dist=dist_coef,markerpos_file="markers1to4.csv")
    try:
        while ~cap.isOpened():
            ok,frame = cap.read()
            nframe = cv2.undistort(frame, K, dist_coef)
            tv = vm.getcamerapose(frame)
            print(tv)
            #print(vm.rvec) #euler angle

    except KeyboardInterrupt:
        print("Finish Program!")
        exit(0)