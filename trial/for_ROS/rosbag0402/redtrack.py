#!/usr/bin/env python
# -*- coding: utf-8 -*
# assuming python2 for ROS

import rospy
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import message_filters # for time syncronizer

import cv2
from cv_bridge import CvBridge, CvBridgeError
import tf
import collections
import math
import numpy as np

import sys
sys.path.append('../../../')
from RedTracker import *
from vmarker import *

def finishtest():
    out1.release()
    out2.release()
    cv2.destroyAllWindows()
    exit(0)

'''
main callback
extract position
'''


def callback(limg, rimg, info, depth):
    print("In the call back")
    sys.stdout.flush()
    
    # try to get image
    try:
        # Convert ROS Disparity Image message to OpenCV2
        limg_data = CvBridge().imgmsg_to_cv2(limg, "8UC3")
        rimg_data = CvBridge().imgmsg_to_cv2(rimg, "8UC3")
        D = info.D
        K = info.K
        K = np.array(K).reshape(3,3)
        D = np.array(D)
        dimg = CvBridge().imgmsg_to_cv2(depth, "32FC1")
    except CvBridgeError, e:
        print(e)
            
    
    #limg_fix = cv2.cvtColor(limg_data,cv2.COLOR_BGR2RGB)# true color?        
    #rimg_fix = cv2.cvtColor(rimg_data,cv2.COLOR_BGR2RGB)# true color?
    limg_fix = limg_data
    rimg_fix = rimg_data
    
    # vm initialize
    if not('vml' in globals()):
        print("Init marker program")
        vml = vmarker(K=K,dist=D,markerpos_file="../roomA_ground_orig.csv")
        vmr = vmarker(K=K,dist=D,markerpos_file="../roomA_ground_orig.csv")
        print(dimg)
        
        # save parameter
        np.savetxt("vm_K.csv",K,delimiter=',')
        np.savetxt("vm_D.csv",D,delimiter=',')
        np.savetxt("dimage.csv",dimg,delimiter=',')
        cv2.imwrite("limage.png",limg_fix)
        cv2.imwrite("rimage.png",rimg_fix)

    # tracker initialize
    if not('ltrack' in globals()):
        print("Init tracker program")
        ltrack = RedTracker(limg_fix.copy(),showimage=0,initialize_with_hand=1,bboxsize=36)
        ltrack.track(limg_fix.copy())
        #rtrack = RedTracker(rimg_fix.copy(),showimage=0,initialize_with_hand=1,bboxsize=36)
        
    
        
    ## for saving videos    
    #out1.write(limg_fix)
    #out2.write(rimg_fix)
    
    draw = limg_fix.copy()

    ## tracking
    ltrack.track(draw)

    ## get camera pose
    vml.getcamerapose(draw)

    ## show camera pose
    if vml.hasCameraPose:
        print(vml.rvecs)
        vml.drawaxis(draw)
    cv2.imshow("pose and track",ltrack.drawrect(draw,ltrack.bbox))
    
    
    pos=ltrack.getpos()
    #print(pos)
    #print(dimg(int(pos[1]),int(pos[0])))
    #sys.stdout.flush()
    
    try:
        k = cv2.waitKey(1)
        if k == 27 :
            finish()
    except KeyboardInterrupt:
        finish()
    

## for video record 
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out1 = cv2.VideoWriter('loutput.avi',fourcc, 30.0, (1280,720))
#out2 = cv2.VideoWriter('routput_.avi',fourcc, 30.0, (1280,720))


def main():
    global rimgs,limgs,dimgs,vml,vmr,ltrack,rtrack


    print("Initialization ...")
    rospy.init_node('zed_vm', anonymous=True)

    image_subl = message_filters.Subscriber("/zed/left/image_raw_color", Image)
    image_subr = message_filters.Subscriber("/zed/right/image_raw_color", Image)
    info_sub = message_filters.Subscriber("/zed/left/camera_info", CameraInfo)
    depth_sub = message_filters.Subscriber("/zed/depth/depth_registered", Image)
    
    ts = message_filters.TimeSynchronizer([image_subl, image_subr, info_sub, depth_sub], 10)
    ts.registerCallback(callback)
    
    

    # spin() simply keeps python from exiting until this node is stopped
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
