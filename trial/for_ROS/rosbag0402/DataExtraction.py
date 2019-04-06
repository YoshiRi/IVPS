import cv2
import numpy as np
import sys

sys.path.append('../../../')
import matplotlib.pyplot as plt


import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from vmarker import vmarker
from RedTracker import RedTracker


# find nearlest
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)

import math
def estimateDepth(pos,depth):
    hei,wid = depth.shape
    xm = int(math.floor(pos[0]))
    ym = int(math.floor(pos[1]))
    xp = int(xm+1)
    yp = int(ym+1)
    assert (xm>0 and ym>0 and xp<wid and yp<hei),"pos index is out of the range!"
    rx = xp-pos[0]
    ry = yp-pos[1]
    return (depth[ym,xm]*(rx+ry)+depth[yp,xm]*(1-ry+rx)+depth[ym,xp]*(1-rx+ry)+depth[yp,xp]*(2-rx-ry))/4

def getXYZfromDepth(pos2d,depth,K,vm):
    Z = estimateDepth(pos2d,depth)
    Xf = Z * pos[0]
    Yf = Z * pos[1]
    tc = vm.tvecs.reshape(3,1)
    Rc,_ = cv2.Rodrigues(vm.rvecs)
    to_c = np.dot(np.linalg.inv(K),np.array([Xf,Yf,Z]).reshape(3,1))            
    tc_g = -np.dot(Rc.T,vm.tvecs)
    pos3d = tc_g+np.dot(Rc.T,to_c)
    return pos3d



if __name__ == '__main__':

    try: 
        bagfile = sys.argv[1]
    except:
        '/home/yoshi/Downloads/firsttry0402.bag'
            

    K = np.loadtxt("vm_K.csv",delimiter=",").reshape(3,3)
    D = np.loadtxt('vm_D.csv',delimiter=",")


    timestamps = []
    timestamps_d = []
    images = []
    dimages = []
    for topic, msg, t in  rosbag.Bag(bagfile).read_messages():
        if topic == '/zed/left/image_raw_color':
            timestamps.append(t.to_sec())
            images.append( CvBridge().imgmsg_to_cv2(msg, "8UC3") )
        elif topic == '/zed/depth/depth_registered':
            timestamps_d.append(t.to_sec())
            dimages.append(CvBridge().imgmsg_to_cv2(msg, "32FC1"))



    left_images = images[5:]
    limg_timestamps = timestamps[5:]
    depth_images = dimages[5:]
    dimg_timestamps = timestamps_d[5:]


    vml = vmarker(K=K,dist=D,markerpos_file="../roomA_ground_orig.csv",showimage=0)
    i = 0
    while not vml.hasCameraPose:
        vml.getcamerapose(left_images[i].copy())
        i += 1
    
    

    tracker_name = "MedianFlow" # with 36 window size, slow
    tracker_name = "KCF" # KCF with 64 window size, fast
    tracker_name = "MOSSE" # MOOSE with 100 pix window size, super fast

    bboxsize = 100# 36 50 64 80 100

    ltrack = RedTracker(left_images[0].copy(),showimage=1,tracker=tracker_name,initialize_with_hand=1,bboxsize=bboxsize) # 36 64
    mask, _,_,_ = ltrack.extractRed(left_images[0].copy())
    cv2.imshow("mask",mask)
    cv2.waitKey(0)

    left_marker_2dpos= []
    left_marker_3dpos= []

    for limg in left_images:
        ltrack.track(limg)
        pos = ltrack.getpos()
        left_marker_2dpos.append(pos)
        left_marker_3dpos.append(vml.getobjpose_1(pos,0.079))
        
    cv2.destroyAllWindows()
    t_plot = np.array(limg_timestamps).reshape(-1,1)
    uv_plot = np.array(left_marker_2dpos).reshape(-1,2)
    xy_plot = np.array(left_marker_3dpos).reshape(-1,2)

    plt.figure(1)

    plt.subplot(121)
    plt.plot(t_plot,uv_plot)
    plt.title("image track")
    plt.subplot(122)
    plt.plot(t_plot,xy_plot)
    plt.title("position track")
    plt.show()

    np.savetxt("results/Expresult_0405_"+tracker_name+str(bboxsize)+".csv",np.concatenate([t_plot,uv_plot,xy_plot],axis=1))
    print("Finish program!")


    left_marker_3dpos_depthmethod= []

    for i in range(len(depth_images)):
        # get 2d pos with nearelst one
        pos = left_marker_2dpos[find_nearest(limg_timestamps,dimg_timestamps[i])]
        xyz = getXYZfromDepth(pos,depth_images[i],K,vml)
        left_marker_3dpos_depthmethod.append(xyz)

    time_xyz = np.array(dimg_timestamps).reshape(-1,1)
    xyz_plot = np.array(left_marker_3dpos_depthmethod).reshape(-1,3)

    plt.figure(3)
    plt.plot(time_xyz,xyz_plot)

    plt.show()

    np.savetxt("results/Expresult_0405_"+"DepthbasedMethod"+".csv",np.concatenate([time_xyz,xyz_plot],axis=1))
