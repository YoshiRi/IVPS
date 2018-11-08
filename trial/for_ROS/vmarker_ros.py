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

def extractRed(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域2
    hsv_min = np.array([150,127,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    # RGB search
    bgr_min = np.array([0,0,120])
    bgr_max = np.array([50,50,255])
    mask3 = cv2.inRange(img,bgr_min, bgr_max)

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
        self.PNPsolved = False
    
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
            self.PNPsolved, self.rvecs, self.tvecs, inliers = cv2.solvePnPRansac(self.objp, self.ccorners, self.K, self.dist)
            self.drawaxis(aruco.drawDetectedMarkers(frame,corners,ids)) # draw origin
            self.R,_ = cv2.Rodrigues(self.rvecs)
            return -np.dot(self.R.T,self.tvecs)

        else:
            if self.PNPsolved:
                self.drawaxis(frame)
                return -np.dot(self.R.T,self.tvecs)
            else:
                cv2.imshow('projected',aruco.drawDetectedMarkers(frame,corners,ids))
                cv2.waitKey(1)
                return []

    def draw(self,img, origin, imgpts):
        corner = tuple(origin[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img

    def drawaxis(self,frame):#30cm cube
        self.axis = np.float32([[0.3,0,0], [0,0.3,0], [0,0,0.3]]).reshape(-1,3)
        self.origin = np.float32([[0,0,0]]).reshape(-1,3)
        imgpts, jac = cv2.projectPoints(self.axis, self.rvecs, self.tvecs, self.K, self.dist)
        imgorgs, _ = cv2.projectPoints(self.origin, self.rvecs, self.tvecs, self.K, self.dist)
        img = self.draw(frame,imgorgs,imgpts)
        cv2.imshow('projected',img)
        cv2.waitKey(1)

    # z: object height
    def getobjpose_1(self,objpts,z):
        self.R,_ = cv2.Rodrigues(self.rvecs)
        pt = cv2.undistortPoints(np.array(objpts).reshape(-1,1,2),self.K,self.dist,P=self.K)
        self.Rt = np.concatenate([self.R,self.tvecs],axis=1)
        # Extraction
        self.P = np.dot(self.K,self.Rt)
        A3 = - np.float32([pt[0,0,0],pt[0,0,1],1]).reshape(3,1) #A1,A2 = self.Rt[:,0],self.Rt[:,1] 
        A4 = self.P[:,2:3]*z+self.P[:,3:4]
        A = np.concatenate([self.P[:,0:2],A3,A4],axis=1)
        U, S, V = np.linalg.svd(A) # use svd to get null space
        vec = V[3]
        X = vec[0]/ vec[3]
        Y = vec[1]/ vec[3]

        return [X,Y]
    
    # assume zero height
    def getobjpose_2(self,objpts):
        # get 3d pose and convert to 2d
        plane2dmap = self.objp[:,0:2].reshape(-1,1,2)
        # then extract homography from 2dpose and image points
        Homo,inliner = cv2.findHomography(self.ccorners,plane2dmap,cv2.RANSAC,3.0)
        # finally from using homography to convert observed pts to 3d pose
        pos=cv2.perspectiveTransform(np.float32([cx,cy]).reshape(-1,1,2),Homo)
        return pos[0,0]
        

if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    # load camera matrix and distort matrix
    K = np.loadtxt("K.csv",delimiter=",")
    dist_coef = np.loadtxt('d.csv',delimiter=",")
    vm = vmarker(K=K,dist=dist_coef,markerpos_file="roomA_ground_orig.csv")
    try:
        while ~cap.isOpened():
            ok,frame = cap.read()
            #nframe = cv2.undistort(frame, K, dist_coef)
            mask,cpts,flag = extractRed(frame)
            cv2.imshow("mask",mask)
            if vm.hasCameraPose*flag:
                objxy = vm.getobjpose_1(cpts,0.13)
                print([objxy[0] ,objxy[1]])
            else:
                tv = vm.getcamerapose(frame)
                #cv2.imwrite('extraction.png',frame)
            vm.showmarker(frame)
            cv2.waitKey(1)
            

    except KeyboardInterrupt:
        print("Finish Program!")
        exit(0)