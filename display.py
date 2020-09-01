import cv2
import numpy as np 


cap = cv2.VideoCapture('test.mp4')

#feautre extractor
def extractfeatures(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,100,0.01,0.5)
    corners = np.int0(corners)
    return corners

#function for matching features with the previous frame

def matcher(frame,f,last):
    bf = cv2.BFMatcher_create()

    matches = None
    orb = cv2.ORB_create(1000)
    if frame is not None:
        kps = orb.detect(frame,None)
        kps, des = orb.compute(frame,kps)
        if last is not None:
            matches = bf.knnMatch(des,last,2)
            print(matches)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            
    
    return kps,des,matches





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
                kps,des,matches = matcher(frame,i,None)
            else:
                kps,des,matches = matcher(frame,i,last['des'])
            last = {'kps':kps,'des':des,'matches':matches}

            cv2.circle(frame,(x,y),3,(0,255,255),-1)
        cv2.imshow('Frame',frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
   
    else:
        break
   

cap.release() 
cv2.destroyAllWindows() 