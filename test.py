import numpy as np
import cv2
e features


MIN_MATCH_COUNT = 10

img1 = cv2.imread("./buttets/test1.jpg")
img2 = cv2.imread("./buttets/test2.jpg") 
img1 = cv2.resize(img1,(800,800))
img2 = cv2.resize(img2,(800,800))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)    
corners = cv2.goodFeaturesToTrack(gray1,10000,0.01,3)
corners2 = cv2.goodFeaturesToTrack(gray2,10000,0.01,3)
kps = []
kps2 = []
for i in corners:
    x,y=i.ravel()
    kpp = cv2.KeyPoint(x,y,20)
    kps.append(kpp)
for i in corners2:
     x,y=i.ravel()
     kpp = cv2.KeyPoint(x,y,20)
     kps2.append(kpp)

detector = cv2.ORB_create(cv2.NORM_HAMMING)
kp1, des1 = detector.compute(gray1, kps)
kp2, des2 = detector.compute(gray2, kps2)

print("keypoints: {}, descriptors: {}".format(len(kp1), len(des1)))
print("keypoints: {}, descriptors: {}".format(len(kp2), len(des2)))   
good = []
ptlists = []
# Match the features
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des2,des1, k=2)

for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        try:
            src = (int(kp1[m.trainIdx].pt[0]),int(kp1[m.trainIdx].pt[1]))
            dst = (int(kp2[m.queryIdx].pt[0]),int(kp2[m.queryIdx].pt[1]))
            ptlists.append([src,dst])
        except:
            pass
for p in ptlists:
    cv2.circle(img2,p[0],3,(255,255,0),1)
    cv2.line(img2,p[0],p[1],(0,0,255),1)
    
cv2.imshow('img',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(len(matches))