#This is for Github
#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import numpy as np
import cv2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage import transform
from skimage import filters
from skimage import feature
from skimage import morphology
from scipy import signal
from os import path
import sys
import os
#from CAIS import *
np.set_printoptions(threshold=50)
from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.filters import threshold_isodata
from imutils import contours as contours_
#import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z
  
def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)
  
def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)
  
def unit(v):
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)
  
def distance(p0,p1):
    return length(vector(p0,p1))
  
def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)
  
def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)
  
def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

def vertical_lines(img2,fold):
    sum_values = []
    y_mean = []
    y_len,x_len=img2.shape    
    for y in range(fold):
         cropped_image=img2[round((y*y_len)/fold):round(((y+1)*y_len)/fold),:]         
         sum_val=cv2.reduce(cropped_image, 0,  cv2.REDUCE_SUM, dtype=cv2.CV_32S)
         sum_values.append([sum_val])
         y_mean.append((round((y*y_len)/fold)+round(((y+1)*y_len)/fold))/2)
    return y_mean,sum_values 

def runs_of_ones_array(bits):
  # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
  # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return np.round((run_ends + run_starts)/2)
  #return np.round(run_ends)

def getImagesInDirectory(dir):
    return list(filter(
        lambda file:
            path.isfile(os.path.join(dir, file)) and
            (
                file.split('.')[-1].upper() == 'JPG' or
                file.split('.')[-1].upper() == 'JP2'
            ),
        os.listdir(dir)
    ))

def table_lines(img2):
    dds      = img2.copy()
    

    blur      = cv2.GaussianBlur(img2.copy(),(7,7),0)
    laplacian = cv2.Laplacian(blur,cv2.CV_8UC1)
#    lines     = cv2.HoughLinesP(laplacian,1,np.pi/360, 2, None, 2,.1 )
     # Vertical lines
    lines = cv2.HoughLinesP(laplacian, 1, np.pi, threshold=100, minLineLength=25, maxLineGap=1) 
    
    dds[:,:] = 255
    for i in range(len(lines)):    
        x1 = lines[i,0,0]
        y1 = lines[i,0,1]
        x2 = lines[i,0,2]
        y2 = lines[i,0,3]
        cv2.line(dds,(x1,y1),(x2,y2),(0,0,255),1)

    #Vertical
    dilStructure         = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    #dilStructure         = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
    dilution             = cv2.erode(255-dds.copy(),dilStructure,iterations = 1)

    blur                 = cv2.GaussianBlur(dilution.copy(),(7,7),0)
    blur                 = cv2.GaussianBlur(dilution.copy(),(3,3),0)
    dilStructure         = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
    dilution             = cv2.dilate(blur,dilStructure,iterations = 1)

    tresh                = threshold_otsu(dilution)
    img,vertical         = cv2.threshold(dilution,tresh,255,0)

   # Horizontal lines
    lines = cv2.HoughLinesP(laplacian, 1, np.pi / 10, threshold=100, minLineLength=5, maxLineGap=1)
    
    dds[:,:] = 255
    for i in range(len(lines)):    
        x1 = lines[i,0,0]
        y1 = lines[i,0,1]
        x2 = lines[i,0,2]
        y2 = lines[i,0,3]
        cv2.line(dds,(x1,y1),(x2,y2),(0,0,255),1)
    #Horizontal
    dilStructure         = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    #dilStructure         = cv2.getStructuringElement(cv2.MORPH_RECT, (30,1))
    dilution             = cv2.erode(255-dds.copy(),dilStructure,iterations = 1)
    
    blur                 = cv2.GaussianBlur(dilution.copy(),(3,3),0)
    #blur                 = cv2.GaussianBlur(blur,(7,7),0)
    #blur                 = cv2.GaussianBlur(blur,(7,7),0)
    dilStructure         = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
    dilution             = cv2.dilate(blur,dilStructure,iterations = 1)

    tresh                = threshold_otsu(dilution)
    img,horizontal       = cv2.threshold(dilution,tresh,255,0)
    
    mask = np.zeros(vertical.shape, np.uint8)
      
    __, ctsk, hierarchy = cv2.findContours(vertical.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    bigC = []
    for c in ctsk:
        x,y,w,h = cv2.boundingRect(c)
        if h<40:
        #if w<20: 
           bigC.append(c)
    
    cv2.drawContours(mask, bigC, -1, (255,255,255), -1)
    

#    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('image', ratio[1],ratio[0])
#    cv2.resizeWindow('image', 50,50)
#    cv2.imshow('image',mask)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    vertical = cv2.subtract(vertical,mask)
    return(vertical,horizontal)

def cropping_GS_images(img2):
    
    img2_vert, img2_horz = table_lines(img2)
    y_len,x_len          = img2.shape       

#Working on the vertical part
    fold                 = 10
    y_m,dgs              = vertical_lines(img2_vert,fold)
    ddf                  = np.asarray(dgs).reshape(fold,x_len)
    ddf1                 = (ddf>0)*1
    bits                 = ddf1[5,:]

    dfg                  = np.concatenate(([0],runs_of_ones_array(bits)))
    dfg_                 = np.where(np.diff(dfg)>200)
    beg                  = dfg[dfg_[0][1]+1]-25
    end                  = dfg[dfg_[0][2]]+25

#Working on the horisontal part: Part 1
    img2_horz_to_vert    = cv2.transpose(img2_horz[:,int(beg):int(end)])
    y_len,x_len          = img2_horz_to_vert.shape

    fold                 = 2
    y_m,dgs              = vertical_lines(img2_horz_to_vert,fold)
    ddf                  = np.asarray(dgs).reshape(fold,x_len)
    ddf1                 = (ddf>0)*1
    bits                 = ddf1[1,:]
    
    dfg                  = np.concatenate(([0],runs_of_ones_array(bits)))
    dfg_                 = np.argmax(np.diff(dfg))
    begh                 = 0
    endh                 = dfg[dfg_]

#Working on the horisontal part: Part 2
    img2_horz_cropped_1  = img2_horz[int(begh):int(endh),int(beg):int(end)]
    y_len,x_len          = img2_horz_cropped_1.shape

    fold                 = 2
    y_m,dgs              =  vertical_lines(img2_horz_cropped_1,fold)
    ddf                  = np.asarray(dgs).reshape(fold,x_len)

    ddf1                 = (ddf>0)*1
    bits                 = ddf1[1,:]
    endhh                = np.max(np.where(np.diff(bits)<0))

    img2_vert_cropped_2  = img2_vert[int(begh):int(endh),int(beg):int(beg)+int(endhh)+25]
    img2_vert_cropped_2[600:615,:]=0   
    img2_vert_cropped_2[1000:1015,:]=0  
    img2_vert_cropped_2[1200:1215,:]=0   
    img2_vert_cropped_2[1400:1415,:]=0  
    img2_vert_cropped_2[1500:1515,:]=0  
    img2_vert_cropped_2[1700:1715,:]=0   
    img2_vert_cropped_2[1800:1815,:]=0   
    
    img2_horz_cropped_2  = img2_horz[int(begh):int(endh),int(beg):int(beg)+int(endhh)+25]
    img2_horz_cropped_2[:,60:65]=0
    img2_horz_cropped_2[:,160:165]=0     
    img2_horz_cropped_2[:,260:265]=0
    img2_horz_cropped_2[:,360:365]=0     
    img2_horz_cropped_2[:,460:465]=0
    img2_horz_cropped_2[:,560:565]=0     
    img2_horz_cropped_2[:,660:665]=0
    img2_horz_cropped_2[:,760:765]=0     
    img2_horz_cropped_2[:,860:865]=0
    img2_horz_cropped_2[:,960:965]=0     
    img2_horz_cropped_2[:,1060:1065]=0     

    y_cropped_range      = np.array((int(begh),int(endh)))
    x_cropped_range      = np.array((int(beg),int(beg)+int(endhh)+25)) 
    
    return y_cropped_range,x_cropped_range,cv2.transpose(img2_vert_cropped_2),img2_horz_cropped_2

def horisontal_lines(v1):
    y_len,x_len = v1.shape 
    v1mask      = v1.copy()
    v1cleansheet = v1.copy()
    v1cleansheet[:,:] = 0

    __, ctsk, hierarchy = cv2.findContours(v1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    bigC = []
    for c in ctsk:
        x,y,w,h = cv2.boundingRect(c)
        #if h>15:
        if x>10:        
            bigC.append(c)
    
    (ctsk_index_sort,_) = contours_.sort_contours(bigC, method="top-to-bottom")        
    cX   =[]
    cY   =[]
    ct   =[]
    ht   =[]
    wt   =[]
    for c in ctsk_index_sort:
        x,y,w,h = cv2.boundingRect(c)
        M       = cv2.moments(c)
        try:
            cX.append(int(M["m10"]/M["m00"]))
            cY.append(int(M["m01"]/M["m00"])) 
            ct.append(c) 
            ht.append(h)
            wt.append(w)
        except:
            pass

    labelcXcY =[]
    m         = 10000
    nblines   = 0
    while len(cX)>2 and nblines <len(ctsk):
        index = np.argsort(wt)[::-1]
        nblines=nblines+1
        [vx,vy,x,y] = cv2.fitLine(ct[index[0]],cv2.DIST_L2,0,0.01,0.01)
        #cv2.line(v1mask, (x-m*vx[0], y-m*vy[0]), (x+m*vx[0], y+m*vy[0]), (255,255,255),10)

        pt  =[]
        xyz =[]
        for i in range(len(cX)):
            ss,ss1 = pnt2line((cX[i],cY[i],0),(x-round(m*vx[0]), y-round(m*vy[0]),0),(x+round(m*vx[0]), y+round(m*vy[0]),0))
            pt.append(ss)
            xyz.append([(cX[i]),(cY[i]),(0)])
        
        izd  = np.where(np.asarray(pt)<5)
        izd  = np.argsort(pt)[:np.min((2,len(izd[0])))]
        xyz_ = np.asarray(xyz)    
        ddd  = xyz_[:,:][izd,:2]    
        
        [vx1,vy1,x1,y1] = cv2.fitLine(ddd,cv2.DIST_L2,0,0.01,0.01)
        #cv2.line(v1mask, (x1-round(m*vx1[0]), y1-round(m*vy1[0])), (x1+round(m*vx1[0]), y1+round(m*vy1[0])), (255,255,255),5)
        #cv2.line(v1mask, (x1-round(m*vx1[0]), (y1-20)-round(m*vy1[0])), (x1+round(m*vx1[0]), (y1-20)+round(m*vy1[0])), (255,255,255),2)
        #cv2.line(v1mask, (x1-round(m*vx1[0]), (y1+20)-round(m*vy1[0])), (x1+round(m*vx1[0]), (y1+20)+round(m*vy1[0])), (255,255,255),2)
        #cv2.line(v1mask,(0,ybot),(x_len,ytop),(255,255,255),1)    
    
        pt  =[]
        xyz =[]
        for i in range(len(cX)):
            ss,ss1 = pnt2line((cX[i],cY[i],0),(x1-round(m*vx1[0]), y1-round(m*vy1[0]),0),(x1+round(m*vx1[0]), y1+round(m*vy1[0]),0))
            pt.append(ss)
            xyz.append([(cX[i]),(cY[i]),(0)])
            
            izd  = np.where(np.asarray(pt)<15)
            izd  = np.argsort(pt)[:np.min((4,len(izd[0])))]
            xyz_ = np.asarray(xyz)    
            ddd  = xyz_[:,:][izd,:2]    
    
        pt  =[]
        xyz =[]
        for i in range(len(cX)):
            ss,ss1 = pnt2line((cX[i],cY[i],0),(x1-round(m*vx1[0]), y1-round(m*vy1[0]),0),(x1+round(m*vx1[0]), y1+round(m*vy1[0]),0))
            pt.append(ss)
            xyz.append([(cX[i]),(cY[i]),(0)])
        
        izd  = np.where(np.asarray(pt)<15)
        izd  = np.argsort(pt)[:np.min((8,len(izd[0])))]
        xyz_ = np.asarray(xyz)    
        ddd  = xyz_[:,:][izd,:2]    

        [vx1,vy1,x1,y1] = cv2.fitLine(ddd,cv2.DIST_L2,0,0.01,0.01)
        
        pt  =[]
        xyz =[]
        for i in range(len(cX)):
            ss,ss1 = pnt2line((cX[i],cY[i],0),(x1-round(m*vx1[0]), y1-round(m*vy1[0]),0),(x1+round(m*vx1[0]), y1+round(m*vy1[0]),0))
            pt.append(ss)
            xyz.append([(cX[i]),(cY[i]),(0)])
        
        izd  = np.where(np.asarray(pt)<25)
        izd  = np.argsort(pt)[:np.min((25,len(izd[0])))]
        xyz_ = np.asarray(xyz)    
        ddd  = xyz_[:,:][izd,:2]    
        
        [vx1,vy1,x1,y1] = cv2.fitLine(ddd,cv2.DIST_L2,0,0.01,0.01)
        
        labelcXcY.append(([cX[i] for i in izd],[cY[i] for i in izd],[nblines for i in izd]))
        
        izd    = np.where(np.asarray(pt)<25)
        cX     = np.delete(cX,izd)
        cY     = np.delete(cY,izd)
        ct     = np.delete(ct,izd)
        ht     = np.delete(ht,izd)
        wt     = np.delete(wt,izd)  
        
        for i in range(nblines):
            if len(labelcXcY[i][0])>2:
                z   = np.polyfit(labelcXcY[i][0],labelcXcY[i][1],1)
                f   = np.poly1d(z)
                xs  = np.linspace(0,m,1000)
                ys  = f(xs)
                ddd = np.vstack((xs,ys)).T
                cv2.polylines(v1cleansheet,np.int32([ddd]),0,(255,255,255),2)
    return v1cleansheet         

def horizontal_sweepline(v1):
    y_len,x_len         = v1.shape 
    v1cleansheet        = v1.copy()
    #v1cleansheet[:,:]   = 0    
    bigC = []
    bigRC =[]
    __, ctsk, hierarchy = cv2.findContours(v1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    for c in ctsk:
            x,y,w,h = cv2.boundingRect(c)
            #if h>15:
            if w>30:        
                bigC.append(c)
                
    (cX,_) = contours_.sort_contours(bigC, method="right-to-left")    
    
    while len(cX)>2:
        (cX,_) = contours_.sort_contours(cX, method="right-to-left") 
        ctsk_sort = cX
        idx = 0
        bigC = []
        idz = []
        for c in ctsk_sort:
            x,y,w,h = cv2.boundingRect(c)
            if idx==0:
                bigC.append((int(x+0.5*w),int(y+0.5*h)))
                idz.append(idx)
                y_up = max(y+20,0)
                y_dw = max(y-20,0)                
            else:
                if y<y_up and y>y_dw:
                    bigC.append((int(x+0.5*w),int(y+0.5*h)))
                    idz.append(idx)
                    y_up = min(y+20,y_len)
                    y_dw = max(y-20,0)    
            idx += 1
        cX = np.delete(cX,idz)    
        bigRC.append(bigC)     
        
    for xx in bigRC:
        if len(xx)>4:
            z   = np.polyfit([xx[i][0] for i in range(len(xx))],[xx[i][1] for i in range(len(xx))],2)
            f   = np.poly1d(z)
            xs  = np.linspace(0,10000,10000)
            ys  = f(xs)
            ddd = np.vstack((xs,ys)).T
            cv2.polylines(v1cleansheet,np.int32([ddd]),0,(255,255,255),2)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('image', ratio[1],ratio[0])
            cv2.resizeWindow('image', 50,50)
            cv2.imshow('image',v1cleansheet)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return v1cleansheet         
