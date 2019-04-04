#!/usr/bin/env python
# -*- coding: utf-8 -*
# assuming python3 

import cv2
import numpy as np


class RedTracker:
    def __init__(self,frame,initialize_with_hand=1,tracker='KCF',showimage=0,bboxsize=20):
        self.bboxsize = bboxsize
        self.refreshTHDtracker = 5  # the maximum acceptable center of mass position error
        self.pos = [] # tracked 2d point

        # if init with hand:
        if initialize_with_hand:
            rect = cv2.selectROI(frame, False)
            roirect = self.extractROI(frame,rect)
            _,crois,_,_ = self.extractRed(roirect)
            #self.bbox = (rect[0]+rect[2]/2-self.bboxsize/2,rect[1]+rect[3]/2-self.bboxsize/2,self.bboxsize,self.bboxsize)
            self.bbox = (crois[0]+rect[0]-self.bboxsize/2,crois[1]+rect[1]-self.bboxsize/2,self.bboxsize,self.bboxsize)
            cv2.destroyAllWindows()
        else:   # if init automatically
            self.bbox,_ = self.find_largest_redzone_rect(frame,bboxsize=self.bboxsize)

        # show rect
        self.showrect(frame,waittime=0)
        cv2.destroyAllWindows()

        # initialize tracker
        self.get_tracker(tracker)
        self.ok = self.boxtracker.init(frame, self.bbox)

        self.showimage = showimage

    def get_tracker(self,name):
        """
        Choose tracker from key word
        see here : https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
        """
        self.boxtracker = {
            'Boosting': cv2.TrackerBoosting_create(),
            'MIL': cv2.TrackerMIL_create(),
            'KCF' : cv2.TrackerKCF_create(),# Opencv Reccomendation
            'TLD' : cv2.TrackerTLD_create(),
            'MedianFlow' : cv2.TrackerMedianFlow_create(), # Fast but has drift
            'MOSSE': cv2.TrackerMOSSE_create(), # Super fast template need to be big
            'GOTURN': cv2.TrackerGOTURN_create() # super slow but accurate: not working in current version
        }.get(name, 0)        

    def extractROI(self,frame,roi):
        return frame[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]


    def drawrect(self,frame,bbox,color=(0,255,0)):
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, color, 2, 1)
        return frame

    def showrect(self,frame,waittime=1):
        # show bounding box
        framewithrect = self.drawrect(frame.copy(),self.bbox)
        cv2.putText(framewithrect, "Press Any Key to Start! Rect is " + str(self.bbox), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow("show bbox",framewithrect)
        cv2.waitKey(waittime)

    ## Function to be called every update 
    def track(self,frame):
        self.ok, self.bbox = self.boxtracker.update(frame)
        # track is succeed:
        if self.ok:
            _,centers,_,validnum = self.extractRed(self.extractROI(frame,self.bbox))
            self.pos = [self.bbox[0]+centers[0],self.bbox[1]+centers[1]]
            self.validpixelnum = validnum
            if self.refreshTHDtracker  <  max(abs(self.bboxsize/2 - centers[0]),abs(self.bboxsize/2 - centers[1])):
                self.bbox = (self.pos[0]-self.bboxsize/2,self.pos[1]-self.bboxsize/2,self.bboxsize,self.bboxsize)
                self.boxtracker.init(frame,self.bbox)
                print('reinit tracker!')

            if self.showimage:
                cv2.imshow("tracked", self.drawrect(frame.copy(),self.bbox))
                cv2.waitKey(1)

        else:
            print("Failed to track!")
            if self.showimage:
                cv2.imshow("tracked", self.drawrect(frame.copy(),self.bbox))
                cv2.waitKey(1)


    def getpos(self):
        return self.pos
    def getrect(self):
        return self.bbox
    def getvalidpixelnumber(self):
        return self.validpixelnum

    def find_largest_redzone_rect(self,image,bboxsize=30):
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
        largest = max(rects, key=(lambda x: x[2] * x[3])) #return maximum rectangle
        centerx = largest[0]+largest[2]/2 
        centery = largest[1]+largest[3]/2
        bbox = (centerx-bboxsize/2,centery-bboxsize/2,bboxsize,bboxsize)
        return bbox, largest
        
    def extractRed(self,image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        mask = np.zeros(h.shape, dtype=np.uint8)
        mask[((h < 10) | (h > 200)) & (s > 128)] = 255

        # get RED size
        validnum = sum(mask.reshape(-1))/255
        hei,wid,_ = image.shape

        Mmt = cv2.moments(mask)
        if Mmt["m00"] != 0:
            cx = Mmt['m10']/Mmt['m00']
            cy = Mmt['m01']/Mmt['m00']
            flag = True
        else:
            cx,cy = wid/2,hei/2
            flag = False
        #print([cx,cy])
        return mask,[cx,cy],flag,validnum




if __name__ == '__main__':
    import sys
    try:
        fname = sys.argv[1]
    except:
        fname = 0

    cap = cv2.VideoCapture(fname)
    ok,frame = cap.read()
    
    if not ok:
        sys.exit(-1)

    tracker = RedTracker(frame,showimage=1,initialize_with_hand=0)

    pos1 = []
    pix = []
    # Start timer
    
    while 1:
        ok,frame = cap.read()
        timer = cv2.getTickCount()  
        if not ok:
            break

        tracker.track(frame)
        
        
        pos1.append(tracker.getpos())
        pix.append(tracker.getvalidpixelnumber())
        
        # show FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        print("FPS: "+str(fps))
        

        k = cv2.waitKey(1)
        if k == 27 :
            break

    import matplotlib.pyplot as plt

    p1 = np.array(pos1).reshape(-1,2)
    pix1 = np.array(pix).reshape(-1)

    plt.figure(1)
    plt.plot(p1[:,0],-p1[:,1],label='Coarse&Fine')
    plt.legend()
    plt.figure(2)
    plt.subplot(121)
    plt.plot(p1[:,0],label='Coarse&Fine x')
    plt.subplot(122)
    plt.plot(p1[:,1],label='Coarse&Fine y')
    plt.legend()
    plt.figure(3)
    plt.plot(pix1,label='number of valid pixel')
    plt.legend()
    plt.show()

    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    cv2.destroyAllWindows()