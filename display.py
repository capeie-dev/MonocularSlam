import cv2
import numpy as np 


cap = cv2.VideoCapture('test.mp4')
index_params = dict(algorithm = 1, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
orb = cv2.ORB_create(500)
bf = cv2.BFMatcher()

#feautre extractor
def extractfeatures(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,100,0.01,0.5)
    corners = np.int0(corners)
    return corners

#function for matching features with the previous frame

def matcher(frame,f,last):
    matches = None
   
    if frame is not None:        
        kps, des = orb.compute(frame,f)
        if last is not None:
            matches = bf.knnMatch(des,last,k=2)
        
        print(matches)
        if matches is not None:
            for m,n in matches:
                print('Its working!')
            
    return kps,des,matches





#This is main, but we at capeie corp belive that main functions are not cool, also frick the lidars
last = None
while(cap.isOpened()):
    ret,frame = cap.read()
    
    
    kpframe = []
    
    if ret == True:
        
        frame = cv2.resize(frame,(540,540))
        corners = extractfeatures(frame)
        for i in corners:
            x,y = i.ravel()
            storer = cv2.KeyPoint(x,y,20)
            kpframe.append(storer)
            

            cv2.circle(frame,(x,y),3,(0,255,255),-1)
        
        if last is None:
            kps,des,matches = matcher(frame,kpframe,None)
            
        else:
            kps,des,matches = matcher(frame,kpframe,last['des'])
        
        last = {'kps':kpframe,'des':des,'matches':matches}
        
        cv2.imshow('Frame',frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
   
    else:
        break
   

cap.release() 
cv2.destroyAllWindows() 