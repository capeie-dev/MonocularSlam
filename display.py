import cv2
import numpy as np 


cap = cv2.VideoCapture('test8.mp4')
index_params = dict(algorithm = 1, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
orb = cv2.ORB_create(100)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

#feautre extractor
def extractfeatures(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,10000,0.01,0.1)
    corners = np.int0(corners)
    return corners

#function for matching features with the previous frame

def matcher(frame,f,last1,last2):
    matches = None
    good = []
    ptlists=[]
    if frame is not None:        
        kps, des = orb.compute(frame,f)
        if last1 is not None:
            matches = bf.knnMatch(des,last1,k=2)
        if matches is not None:
           
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
                  
                    src = (int(last2[m.trainIdx].pt[0]),int(last2[m.trainIdx].pt[1]))
                    dst = (int(kps[m.queryIdx].pt[0]),int(kps[m.queryIdx].pt[1]))
            
                    ptlists.append([src,dst])
                    
    
    return kps,des,matches,ptlists



#This is main, but we at capeie corp belive that main functions are not cool, also frick the lidars
last = None
good = None
while(cap.isOpened()):
    ret,frame = cap.read()
    kpframe = []
    
    if ret == True:
        
        frame = cv2.resize(frame,(800,800))
        corners = extractfeatures(frame)
        for i in corners:
            x,y = i.ravel()
            storer = cv2.KeyPoint(x,y,20)
            kpframe.append(storer)
            
        
        if last is None:
            kps,des,matches,ptlists= matcher(frame,kpframe,None,None)
            
        else:
            kps,des,matches,ptlists = matcher(frame,kpframe,last['des'],last['kps'])    
            
            for p in kps:
                cv2.circle(frame,(int(p.pt[0]),int(p.pt[1])),3,(100,100,0),1)
            for p in ptlists:
                cv2.circle(frame,p[0],3,(255,255,0),1)
                cv2.line(frame,p[0],p[1],(0,0,255),1)   

        last = {'kps':kps,'des':des,'matches':matches,'frame':frame,'good':ptlists}

        
        cv2.imshow('Frame',frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
   
    else:
        break
   

cap.release() 
cv2.destroyAllWindows() 