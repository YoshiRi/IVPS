import sys
import math
import matplotlib.pyplot as plt
sys.path.append("../../")
sys.path.append("../")

from vmarker import *
from roomA_video import extractRed
K = np.loadtxt("../../calib_usb/K.csv",delimiter=",")
dist_coef = np.loadtxt('../../calib_usb/d.csv',delimiter=",")




if __name__=='__main__':

    if len(sys.argv) == 1:
        capname = 'x-05y-05'
    else:
        capname = sys.argv[1]

    print('Read '+capname)
    cap = cv2.VideoCapture(capname+'.avi')
    vm = vmarker(K=K,dist=dist_coef,markerpos_file="../roomA_ground_orig.csv")
    rvs = []
    tvs = []

    while 1:
        ok,frame = cap.read()
        if not ok:
            print("Finished!")
            break
        sframe = frame.copy()
        cv2.startWindowThread()
        tv=vm.getcamerapose(frame)
        cv2.waitKey(1)
        if vm.PNPsolved:
            rvs.append(vm.rvecs)
            tvs.append(vm.tvecs)
            

    tvplot = np.array(tvs).reshape(-1,3)
    rvplot = np.array(rvs).reshape(-1,3)

    plen = len(tvplot)
    mtv = np.mean(tvplot,axis=0)
    mrv = np.mean(rvplot,axis=0)

    success_flag = vm.hasCameraPose
    if success_flag:
        fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
        plt.subplot(231)
        plt.plot(tvplot[:,0])
        plt.plot([0,plen],[mtv[0], mtv[0]])
        #plt.legend("tx data","mean")
        plt.subplot(232)
        plt.plot(tvplot[:,1])
        plt.plot([0,plen],[mtv[1], mtv[1]])
        #plt.legend("ty data","mean")
        plt.subplot(233)
        plt.plot(tvplot[:,2])
        plt.plot([0,plen],[mtv[2], mtv[2]])
        #plt.legend("tz data","mean")
        plt.subplot(234)
        plt.plot(rvplot[:,0])
        plt.plot([0,plen],[mrv[0], mrv[0]])
        #plt.legend("rx data","mean")
        plt.subplot(235)
        plt.plot(rvplot[:,1])
        plt.plot([0,plen],[mrv[1], mrv[1]])
        #plt.legend("ry data","mean")
        plt.subplot(236)
        plt.plot(rvplot[:,2])
        plt.plot([0,plen],[mrv[2], mrv[2]])
        #plt.legend("rz data","mean")
        #plt.ion()
        plt.show()
 
    # ture camera poses
    Rmean,_ = cv2.Rodrigues(np.float32(mrv))
    print(Rmean)
    tmean = -np.dot(Rmean.T,np.float32(mtv))
    print(tmean)

    cap = cv2.VideoCapture(capname+'.avi')
    vm = vmarker(K=K,dist=dist_coef,markerpos_file="../roomA_ground_orig.csv")

    centerpts = []
    while 1:
        ok,frame = cap.read()
        if not ok:
            print("Finished!")
            break
        sframe = frame.copy()
        #cv2.startWindowThread()
        mask,cpts,flag = extractRed(frame)
        if flag:
            centerpts.append(cpts)

    plt.figure(2)
    plt.plot(centerpts)
    #plt.ion()
    plt.show()


    if success_flag:
        vm.rvecs = np.array(mrv)
        vm.tvecs = np.array(mtv).reshape(3,1)
    else:
        print("SET PRESET VALUE")
        vm.rvecs = np.array([1.72836771043141, 1.2681418485393647, -0.6854996049482])
        vm.tvecs = np.array([0.2619710716591172, 0.29167728232950163, 3.457404829379607]).reshape(3,1)
        

    objpts = []
    for points in centerpts:
        objxy = vm.getobjpose_1(points,0.13)
        #print([objxy[0] ,objxy[1]])
        objpts.append(objxy)

    plt.figure(3)
    plt.plot(objpts)
    mobjpts = np.mean(objpts,axis=0)
    print(mobjpts)
    plt.show()

    # save result with yaml
    import yaml
    with open(capname+'.yml', 'w') as stream:
        yaml.dump({"rvecs":mrv.tolist()},stream)
        yaml.dump({"tvecs":mtv.tolist()},stream)
        yaml.dump({"objpts":mobjpts.tolist()},stream)
        yaml.dump({"imgpts":centerpts},stream)
        yaml.dump({"rvdata":rvs},stream)
        yaml.dump({"tvdata":tvs},stream)

    
    