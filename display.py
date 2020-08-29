import cv2
import numpy as np 


cap = cv2.VideoCapture('test.mp4')

#feautre extractor
def extractfeatures(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,1500,0.01,0.5)
    corners = np.int0(corners)
    return corners

#function for matching features with the previous frame

def matcher(frame,f,last):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    orb = cv2.ORB_create(500)
    if frame is not None:

        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1],_size=20)]
        kps, des = orb.compute(frame,kps)
        if last is not None:
            matches = bf.match(des,last)
            print(matches)


    return kps,des





#This is main, but we at capeie corp belive that main functions are not cool, also frick the lidars

while(cap.isOpened()):
    ret,frame = cap.read()
    last = None

    if ret == True:
        
        frame = cv2.resize(frame,(540,540))
        corners = extractfeatures(frame)
        for i in corners:
            x,y = i.ravel()
            if last is None:
                kps,des = matcher(frame,i,None)
            else:
                kps,des = matcher(frame,i,last[1])
            last = kps,des

            cv2.circle(frame,(x,y),3,(0,255,255),-1)
        cv2.imshow('Frame',frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
   
    else:
        break
   

cap.release() 
cv2.destroyAllWindows() 