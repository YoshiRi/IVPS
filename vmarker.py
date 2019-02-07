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
        self.hasCameraPose = False
        self.detectparam = aruco.DetectorParameters_create()
        self.detectparam.adaptiveThreshConstant = 5.0
        self.detectparam.cornerRefinementMethod = 0
        #self.detectparam.minMarkerDistanceRate = 0.02
        #self.detectparam.errorCorrectionRate = 0.8
    
    def setmarker(self,fname):
        #self.objp = np.zeros((markernum,3), np.float32)
        self.objp = np.loadtxt(fname,delimiter=",")
        self.mnum , _ = self.objp.shape
        print(self.mnum)
    
    def startvideo(self,vnum=0):
        self.cap = cv2.VideoCapture(0)
    
    def showmarker(self,frame):
        aruco = cv2.aruco
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, self.dictionary, parameters = self.detectparam)
        detect = aruco.drawDetectedMarkers(frame,corners)
        cv2.imshow("detected",detect)
        cv2.waitKey(1)
    
    def load_camerapose_yml(self,file):
        try:
            import yaml
            with open(file, 'r+') as stream:
                dic = yaml.load(stream)
                print(dic)
                self.rvecs = np.float32(dic["rvecs"]).reshape(3,1)
                self.tvecs = np.float32(dic["tvecs"]).reshape(3,1)
                self.PNPsolved = True
        except:
            print("Something going wrong!")
        
    def getcamerapose(self,frame,allow3pts = False,rvec_init = [], tvec_init = []):
        aruco = cv2.aruco
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, self.dictionary, parameters = self.detectparam)
        self.PNPsolved = False
        
        # if corner number is larger than 3
        if len(corners) >= 4:
            # sort based on IDs and use center value
            centercorners = []
            geometrypositions = []
            matched = 0
            for id_,corner in sorted(zip(ids,corners)): #corner=[x11,y11]...
                if id_ > self.mnum:
                    break
                matched += 1
                centercorners.append(np.average(corner,1))
                geometrypositions.append(self.objp[id_])
                
            self.ccorners = np.array(centercorners).reshape(matched,1,2)
            self.realcornerpos = np.array(geometrypositions).reshape(matched,3)
            #print(self.ccorners)

            # Find the rotation and translation vectors.
            self.PNPsolved, self.rvecs, self.tvecs, inliers = cv2.solvePnPRansac(self.realcornerpos, self.ccorners, self.K, self.dist)
            self.hasCameraPose = True
            self.drawaxis(aruco.drawDetectedMarkers(frame,corners,ids)) # draw origin
            self.R,_ = cv2.Rodrigues(self.rvecs)
            return -np.dot(self.R.T,self.tvecs)

        else:
            if self.hasCameraPose:
                self.drawaxis(frame)
                return -np.dot(self.R.T,self.tvecs)
            elif allow3pts and len(corners)==3:
                # sort based on IDs and use center value
                centercorners = []
                geometrypositions = []
                for id_,corner in sorted(zip(ids,corners)): #corner=[x11,y11]...
                    centercorners.append(np.average(corner,1))
                    geometrypositions.append(self.objp[id_])
                    
                self.ccorners = np.array(centercorners).reshape(len(ids),1,2)
                self.realcornerpos = np.array(geometrypositions).reshape(len(ids),3)
                #print(self.ccorners)
                
                rvec_init = np.float32(rvec_init).reshape(3,1)
                tvec_init = np.float32(tvec_init).reshape(3,1)
                
                # Find the rotation and translation vectors.
                self.PNPsolved, self.rvecs, self.tvecs = cv2.solvePnP(self.realcornerpos, self.ccorners, self.K, self.dist, flags = cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess = 1,rvec=rvec_init,tvec=tvec_init)
                self.hasCameraPose = True
                self.drawaxis(aruco.drawDetectedMarkers(frame,corners,ids)) # draw origin
                self.R,_ = cv2.Rodrigues(self.rvecs)
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
        try:
            U, S, V = np.linalg.svd(np.dot(A.T,A)) # use svd to get null space
        except: # see here http://oppython.hatenablog.com/entry/2014/01/21/003245
            S2,vt = np.linalg.eigh(np.dot(A.T ,A)) # if numpy method is unstable
            vt=vt[:,::-1]#,S2 = w[::-1]
            V = vt.T                
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
    import sys
    argv = sys.argv
    if argv == 1:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(argv[1])
    # load camera matrix and distort matrix
    K = np.loadtxt("calib_usb/K.csv",delimiter=",")
    dist_coef = np.loadtxt('calib_usb/d.csv',delimiter=",")
    vm = vmarker(K=K,dist=dist_coef,markerpos_file="sample/data/roomA_ground_orig.csv")
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

    exit(0)