import cv2
import numpy as np 

cap = cv2.VideoCapture('test.mp4')

def extractfeatures(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,500,0.01,10)
    corners = np.int0(corners)

    return corners
while(cap.isOpened()):
    ret,frame = cap.read()


    if ret == True:
        
        frame = cv2.resize(frame,(800,540))
        corners = extractfeatures(frame)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(frame,(x,y),3,(0,255,255),-1)
        cv2.imshow('Frame',frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
   
    else:
        break
   

cap.release() 
cv2.destroyAllWindows() 