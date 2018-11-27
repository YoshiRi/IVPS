# See here:https://ensekitt.hatenablog.com/entry/2017/12/21/200000

import cv2
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

pos1 = []
pos2 = []
pos3 = []

def find_rect_of_target_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    mask = np.zeros(h.shape, dtype=np.uint8)
    mask[((h < 15) | (h > 200)) & (s > 128)] = 255
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    
    for contour in contours:
        approx = cv2.convexHull(contour)
        rect = cv2.boundingRect(approx)
        rects.append(np.array(rect))
    return max(rects, key=(lambda x: x[2] * x[3])) #return maximum rectangle

def extractRed2(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    mask = np.zeros(h.shape, dtype=np.uint8)
    mask[((h < 10) | (h > 200)) & (s > 128)] = 255
    Mmt = cv2.moments(mask)
    if Mmt["m00"] != 0:
        cx = Mmt['m10']/Mmt['m00']
        cy = Mmt['m01']/Mmt['m00']
        flag = True
    else:
        cx,cy = 0,0
        flag = False
    #print([cx,cy])
    return mask,[cx,cy],flag



def extractRed(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    rcenter = 165
    rwid = 15
    dark = 127
    hei,wid,_ = img.shape
    
    for i in range(5):
        # 赤色のHSVの値域2
        hsv_min = np.array([rcenter-rwid,dark,0])
        hsv_max = np.array([rcenter+rwid,255,255])
        mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

        if sum(mask2.reshape(-1))/hei/wid/255 > 0.2:
            break
        
        dark = dark - 10
        rwid = rwid + 5
        # RGB search
        #bgr_min = np.array([0,0,120])
        #bgr_max = np.array([50,50,255])
        #mask3 = cv2.inRange(img,bgr_min, bgr_max)

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

def get_tracker(name):
    """
    キーワードに応じてTrackerを選択
    """
    return {
        'Boosting': cv2.TrackerBoosting_create(),
        'MIL': cv2.TrackerMIL_create(),
        'KCF' : cv2.TrackerKCF_create(),
        'TLD' : cv2.TrackerTLD_create(),
        'MedianFlow' : cv2.TrackerMedianFlow_create()
    }.get(name, 0)  

def extractROI(frame,roi):
    return frame[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]

def drawrect(frame,bbox,color=(0,255,0)):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, color, 2, 1)


if __name__ == '__main__':
    """
    引数１：ファイル名，引数２：Method名，引数３：BoundingBox
    
    Method: Boosting, MIL, KCF, TLD, MedianFlow
    """
    ##---------- File name -----------##
    try:
        filename = sys.argv[1]
    except:
        filename = 'Videos/square50.avi'

    ##---------- Method ----------------##
    try:
        method = sys.argv[2]
    except:
        method = 'KCF'
    print('USE '+method+' Tracker!')
    ##----------  Tuple --------------- ##
    try:
        bbox = eval(sys.argv[3])
        print(bbox)
        hasframe = 1
    except:
        hasframe = 0

    ##----------  Tuple --------------- ##
    try:
        with open(sys.argv[4], mode='rb') as f:
            tracker = pickle.load(f)
        hastracker = 1 
    except:
        hastracker = 0

    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    if not ret:
        print('Can not open file:' + filename +'!')
        exit(-1)

    # capture the frame
    if not hasframe:
        bbox = cv2.selectROI(frame, False)
        print(bbox)
        cv2.destroyAllWindows()
    
    # Load tracker
    if not hastracker:
        tracker = get_tracker(method)
        ok = tracker.init(frame, bbox)
        with open(method+'tracker.pickle', mode='wb') as f:
            pickle.dump(tracker, f)

    

    while True:
        # VideoCaptureから1フレーム読み込む
        ret, frame = cap.read()
        if not ret:
            break

        # Start timer
        timer = cv2.getTickCount()

        # トラッカーをアップデートする
        track, bbox = tracker.update(frame)

        cv2.imshow('tracked',extractROI(frame,bbox))
        rmask,centers,_ = extractRed2(extractROI(frame,bbox))
        rmask_ = cv2.circle(rmask, (int(centers[0]), int(centers[1])), 1, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.imshow('tracked_mask',rmask_)
        pos1.append([bbox[0]+centers[0],bbox[1]+centers[1]])

        mask,centers,_ = extractRed2(frame)
        pos3.append([centers[0],centers[1]])
        cv2.imshow('raw_mask',mask)
        

        # FPSを計算する
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 検出した場所に四角を書く
        if track:
            # Tracking success
            redbox = find_rect_of_target_color(frame)
            pos2.append([redbox[0]+redbox[2]/2,redbox[1]+redbox[3]/2])
            drawrect(frame,redbox)
            drawrect(frame,bbox)

        else :
            # トラッキングが外れたら警告を表示する
            cv2.putText(frame, "Failure", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        # FPSを表示する
        cv2.putText(frame, "FPS : " + str(int(fps)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        

        # 加工済の画像を表示する
        cv2.imshow("Tracking", frame)

        # キー入力を1ms待って、k が27（ESC）だったらBreakする
        k = cv2.waitKey(1)
        if k == 27 :
            break


p1 = np.array(pos1).reshape(-1,2)
p2 = np.array(pos2).reshape(-1,2)
p3 = np.array(pos3).reshape(-1,2)
plt.figure(1)
plt.plot(p1[:,0],-p1[:,1],p2[:,0],-p2[:,1],p3[:,0],-p3[:,1])
plt.legend(['Coarse&Fine','Fine','Rect'])

plt.figure(2)
plt.subplot(121)
plt.plot(p1[:,0])
plt.plot(p2[:,0])
plt.plot(p3[:,0])
plt.legend(['Coarse&Fine','Fine','Rect'])
plt.subplot(122)
plt.plot(p1[:,1])
plt.plot(p2[:,1])
plt.plot(p3[:,1])
plt.legend(['Coarse&Fine','Fine','Rect'])
plt.show()

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()