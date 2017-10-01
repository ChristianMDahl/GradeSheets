# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:58:19 2017

@author: cmd
"""

import numpy as np
import opencv
from imutils import contours as contours_

img2 = cv2.imread('croppedhorizontalDSCN3890.JPG',0)
y_len,x_len = img2.shape 
bigC = []
__, ctsk, hierarchy = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
for c in ctsk:
        x,y,w,h = cv2.boundingRect(c)
        #if h>15:
        if h>10:        
            bigC.append(c)
            
(cX,_) = contours_.sort_contours(bigC, method="left-to-right")   
bigRC =[]

while len(cX)>2:
    idx = 0
    ctsk_sort=cX
    bigC = []
    idz = []
    print(idx)
    for c in ctsk_sort:
        x,y,w,h = cv2.boundingRect(c)
        if idx==0:
            bigC.append((x,y))
            idz.append(idx)
            y_up = max(y+h+30,0)
            y_dw = max(y+h-30,0)
        else:
            if y<y_up and y>y_dw:   
                print(idx)
                bigC.append((x,y))
                idz.append(idx)
                y_up = min(y+h+20,y_len)
                y_dw = max(y+h-20,0)                
        idx += 1
    cX = np.delete(cX,idz)    
    bigRC.append(bigC)     
    
 
m= 10000
for xx in bigRC:
    [vx1,vy1,x1,y1] = cv2.fitLine(np.asarray(xx),cv2.DIST_L2,0,0.01,0.01)
    cv2.line(img2, (x1-round(m*vx1[0]), y1-round(m*vy1[0])), (x1+round(m*vx1[0]), y1+round(m*vy1[0])), (255,255,255),5)
        #cv2.line(v1mask, (x1-round(m*vx1[0]), (y1-20)-round(m*vy1[0])), (x1+round(m*vx1[0]), (y1-20)+round(m*vy1[0])), (255,255,255),2)
        #cv2.line(v1mask, (x1-round(m*vx1[0]), (y1+20)-round(m*vy1[0])), (x1+round(m*vx1[0]), (y1+20)+round(m*vy1[0])), (255,255,255),2)
