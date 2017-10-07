# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:58:19 2017

@author: cmd
"""

import numpy as np
import opencv
from imutils import contours as contours_

img2 = cv2.imread('croppedhorizontalDSCN3890.JPG',0)
img2[:,60:65]=0
img2[:,160:165]=0     
img2[:,260:265]=0
img2[:,360:365]=0     
img2[:,460:465]=0
img2[:,560:565]=0     
img2[:,660:665]=0
img2[:,760:765]=0     
img2[:,860:865]=0
img2[:,960:965]=0     
img2[:,1060:1065]=0    

y_len,x_len = img2.shape 
bigC = []
__, ctsk, hierarchy = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
for c in ctsk:
        x,y,w,h = cv2.boundingRect(c)
        #if h>15:
        if w>25:        
            bigC.append(c)
            
(cX,_) = contours_.sort_contours(bigC, method="left-to-right")   
bigRC =[]

while len(cX)>4:
    (cX,_) = contours_.sort_contours(cX, method="left-to-right") 
    ctsk_sort = cX
    idx = 0
    bigC = []
    idz = []
    print(idx)
    for c in ctsk_sort:
        x,y,w,h = cv2.boundingRect(c)
        M       = cv2.moments(c)
        if idx==0:
            bigC.append((x,y))
            bigC.append((x+w,y+h))
            try:
             bigC.append((int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])))  
            except:
             pass 
            idz.append(idx)
            y_up = max(y+20,0)
            y_dw = max(y-20,0)
            
            x_p = x+w
            y_p = y
        else:
            if y<y_up and y>y_dw:     
                cv2.line(img2, (x_p,y_p), (x,y), (255,255,255),5)
                bigC.append((x,y))
                bigC.append((x+w,y+h))                
                try:
                  bigC.append((int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])))  
                except:
                  pass 
                idz.append(idx)
                y_up = min(y+20,y_len)
                y_dw = max(y-20,0)                                
                x_p = x+w
                y_p = y
        idx += 1
    cX = np.delete(cX,idz)    
    bigRC.append(bigC)     
    
 
m= 10000
for xx in bigRC:
    if len(xx)>20:
        print(len(xx))
        #[vx1,vy1,x1,y1] = cv2.fitLine(np.asarray(xx),cv2.DIST_L2,0,0.01,0.01)
        #cv2.line(img2, (x1-round(m*vx1[0]), y1-round(m*vy1[0])), (x1+round(m*vx1[0]), y1+round(m*vy1[0])), (255,255,255),2)
          
        #[vx,vy,x,y]                             = cv2.fitLine(np.asarray(xx), cv2.DIST_L2,0,0.01,0.01)
        #lefty                                   = int((-x*vy/vx) + y)
        #righty                                  = int(((x_len-x)*vy/vx)+y)
        
        #cv2.line(img2, (x_len-1,righty), (0,lefty), (255,255,255),5)
        #cv2.line(v1mask, (x1-round(m*vx1[0]), (y1-20)-round(m*vy1[0])), (x1+round(m*vx1[0]), (y1-20)+round(m*vy1[0])), (255,255,255),2)
        #cv2.line(v1mask, (x1-round(m*vx1[0]), (y1+20)-round(m*vy1[0])), (x1+round(m*vx1[0]), (y1+20)+round(m*vy1[0])), (255,255,255),2)
        z   = np.polyfit([xx[i][0] for i in range(len(xx))],[xx[i][1] for i in range(len(xx))],2)
        f   = np.poly1d(z)
        xs  = np.linspace(0,m,1000)
        ys  = f(xs)
        ddd = np.vstack((xs,ys)).T
        #cv2.polylines(img2,np.int32([ddd]),0,(255,255,255),5)
