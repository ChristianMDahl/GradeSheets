import numpy as np
import cv2
import argparse
import math
import matplotlib.pyplot as plt
import pandas as pd

mkdir                                   = 'Z:\\faellesmappe\\cew\\Sweepline\\SweeplineStraight\\'
#mkdir = 'D:\\CMD\\Dropbox\\CEWCMDShared\\MartinKarlsson\\'
verti                                   = 'croppedverticalDSCN4045.JPG'
hori                                    = 'croppedhorizontalDSCN4045.JPG'
#i                                      = 'croppedverticalDSCN4045.JPG'
imageHori                               = cv2.imread(mkdir + hori,0)
imageVerti                              = cv2.imread(mkdir + verti,0)

blackCanvasHori                         = np.zeros(imageHori.shape)
blackCanvasVerti                        = np.zeros(imageVerti.shape)

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    
    if method == "top-to-bottom":
        i = 1
    
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                key=lambda b:b[1][i], reverse = reverse))
    
    return (cnts, boundingBoxes)

def order_check(image, c, i):
    
    M = cv2.moments(c)
    
    if M["m00"] != 0:
         
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
    else:
        
        cX,cY = 0, 0
    
    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)
    
    return image

def imagePrep(image, picType):
    
    if picType == "horizontal":
    
        _, cntsi, hierarchy                     = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask                                    = np.ones(image.shape[:2], dtype= "uint8")*255

        for a in cntsi:
            x,y,w,h = cv2.boundingRect(a)
            if w < 70:
                cv2.drawContours(mask, [a], -1, 0, -1)
            
        prepped                                 = cv2.bitwise_and(image, image, mask=mask)  

        _, cnts, hierarchy                      = cv2.findContours(prepped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        (cnts, boundingBoxes)                   = sort_contours(cnts, method="left-to-right")
        
        bigCnts                                 = []
        
        for a in cnts:
            bigCnts.append(a)

                
    elif picType == "vertical":
        _, cntsi, hierarchy                     = cv2.findContours(imageVerti.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask                                    = np.ones(imageVerti.shape[:2], dtype= "uint8")*255

        for a in cntsi:
            x,y,w,h = cv2.boundingRect(a)
            if h < 40:
                cv2.drawContours(mask, [a], -1, 0, -1)
                
        prepped = cv2.bitwise_and(imageVerti,imageVerti, mask=mask)
        
        _, cnts, hierarchy                      = cv2.findContours(prepped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        (cnts, boundingBoxes)                   = sort_contours(cnts, method="top-to-bottom")
        
        bigCnts                                 = []
        
        for a in cnts:
            bigCnts.append(a)
            
    return prepped, bigCnts

def sweepline(contourLoop, image, picType):
    
    if picType == "horizontal":
        
        startContour                            = contourLoop
    
        [vx,vy,x,y]                             = cv2.fitLine(startContour, cv2.DIST_L2,0,0.01,0.01)
        rows, cols                              = image.shape[:2]
        lefty                                   = int((-x*vy/vx) + y)
        righty                                  = int(((cols-x)*vy/vx)+y)
        mask                                    = np.zeros(image.shape[:2], dtype= "uint8")
        polypts                                 = np.array([[0,lefty+15],[cols-1,righty+15],[cols-1,righty-15],[0,lefty-15]], np.int32)
        polygon                                 = cv2.fillPoly(mask, pts = [polypts], color=(255,255,255))
        maskFit                                 = cv2.bitwise_and(image,image,mask=polygon)
    
        _, cnts, hierarchy                      = cv2.findContours(maskFit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cnts, boundingBoxes)                   = sort_contours(cnts, method="left-to-right")
        collected                               = np.concatenate((cnts[0:2]),axis=0)
        [vx,vy,x,y]                             = cv2.fitLine(collected, cv2.DIST_L2,0,0.01,0.01)
        lefty                                   = int((-x*vy/vx) + y)
        righty                                  = int(((cols-x)*vy/vx)+y)
        mask                                    = np.zeros(image.shape[:2], dtype= "uint8")
        polypts                                 = np.array([[0,lefty+15],[cols-1,righty+15],[cols-1,righty-15],[0,lefty-15]], np.int32)
        polygon                                 = cv2.fillPoly(mask, pts = [polypts], color=(255,255,255))
        maskFitOne                              = cv2.bitwise_and(image, image, mask=polygon)
    
        _, cnts, hierarchy                      = cv2.findContours(maskFitOne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cnts, boundingBoxes)                   = sort_contours(cnts, method="left-to-right")
        collected                               = np.concatenate((cnts[0:6]),axis=0)
        [vx,vy,x,y]                             = cv2.fitLine(collected, cv2.DIST_L2,0,0.01,0.01)
        lefty                                   = int((-x*vy/vx) + y)
        righty                                  = int(((cols-x)*vy/vx)+y)
        mask                                    = np.zeros(image.shape[:2], dtype= "uint8")
        polypts                                 = np.array([[0,lefty+15],[cols-1,righty+15],[cols-1,righty-15],[0,lefty-15]], np.int32)
        polygon                                 = cv2.fillPoly(mask, pts = [polypts], color=(255,255,255))
        maskFitTwo                              = cv2.bitwise_and(image, image, mask=polygon)
        
        erode                                   = cv2.erode(maskFitTwo, (1,5), iterations = 2)
    
        _, cnts, hierarchy                      = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cnts, boundingBoxes)                   = sort_contours(cnts, method="left-to-right")
        polyLines                               = np.concatenate((cnts[:]),axis=0)
        
        [vx,vy,x,y]                             = cv2.fitLine(polyLines, cv2.DIST_L2,0,0.01,0.01)
        lefty                                   = int((-x*vy/vx) + y)
        righty                                  = int(((cols-x)*vy/vx)+y)
        
        draw                                    = cv2.line(blackCanvasHori, (cols-1,righty), (0,lefty), (255,255,255),5)
        
    elif picType == "vertical":
        
        startContour                            = contourLoop
        
        [vx,vy,x,y]                             = cv2.fitLine(startContour, cv2.DIST_L2,0,0.01,0.01)
        rows, cols                              = image.shape[:2]      
        leftyVerti                              = int((-y*vx/vy) + x)
        rightyVerti                             = int(((rows-y)*vx/vy)+x)
        
        mask                                    = np.zeros(image.shape[:2], dtype= "uint8")
        polypts                                 = np.array([[leftyVerti+15,0],[rightyVerti+15,rows-1],[rightyVerti-15,rows-1],[leftyVerti-15,0]], np.int32)
        polygon                                 = cv2.fillPoly(mask, pts = [polypts], color=(255,255,255))
        maskFit                                 = cv2.bitwise_and(image, image, mask=polygon)
        
        _, cnts, hierarchy                      = cv2.findContours(maskFit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cnts, boundingBoxes)                   = sort_contours(cnts, method="top-to-bottom")
        collected                               = np.concatenate((cnts[0:2]),axis=0)
        
        [vx,vy,x,y]                             = cv2.fitLine(collected, cv2.DIST_L2,0,0.01,0.01)   
        leftyVerti                              = int((-y*vx/vy) + x)
        rightyVerti                             = int(((rows-y)*vx/vy)+x)
        
        mask                                    = np.zeros(image.shape[:2], dtype= "uint8")
        polypts                                 = np.array([[leftyVerti+15,0],[rightyVerti+15,rows-1],[rightyVerti-15,rows-1],[leftyVerti-15,0]], np.int32)
        polygon                                 = cv2.fillPoly(mask, pts = [polypts], color=(255,255,255))
        
        maskFitOne                              = cv2.bitwise_and(image,image,mask=polygon)
        
        _, cnts, hierarchy                      = cv2.findContours(maskFitOne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cnts, boundingBoxes)                   = sort_contours(cnts, method="top-to-bottom")
        collected                               = np.concatenate((cnts[0:6]),axis=0)
        
        [vx,vy,x,y]                             = cv2.fitLine(collected, cv2.DIST_L2,0,0.01,0.01)   
        leftyVerti                              = int((-y*vx/vy) + x)
        rightyVerti                             = int(((rows-y)*vx/vy)+x)
        
        mask                                    = np.zeros(image.shape[:2], dtype= "uint8")
        polypts                                 = np.array([[leftyVerti+15,0],[rightyVerti+15,rows-1],[rightyVerti-15,rows-1],[leftyVerti-15,0]], np.int32)
        polygon                                 = cv2.fillPoly(mask, pts = [polypts], color=(255,255,255))
        
        maskFitTwo                              = cv2.bitwise_and(image,image,mask=polygon)
        
        _, cnts, hierarchy                      = cv2.findContours(maskFitTwo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cnts, boundingBoxes)                   = sort_contours(cnts, method="top-to-bottom")
        collected                               = np.concatenate((cnts[:]),axis=0)
        
        [vx,vy,x,y]                             = cv2.fitLine(collected, cv2.DIST_L2,0,0.01,0.01)   
        leftyVerti                              = int((-y*vx/vy) + x)
        rightyVerti                             = int(((rows-y)*vx/vy)+x)
        
        mask                                    = np.zeros(image.shape[:2], dtype= "uint8") 
        polypts                                 = np.array([[leftyVerti+15,0],[rightyVerti+15,rows-1],[rightyVerti-15,rows-1],[leftyVerti-15,0]], np.int32)
        polygon                                 = cv2.fillPoly(mask, pts = [polypts], color=(255,255,255))
        
        maskFitThree                            = cv2.bitwise_and(image,image,mask=polygon)  

        draw                                 = cv2.line(blackCanvasVerti, (rightyVerti+20,rows-1), (leftyVerti+20,0), (255,255,255), 5)   
        
    return draw

vertiPrep, cnts = imagePrep(imageVerti, picType="vertical")

for a in np.arange(0,50):
    sweepline(cnts[a], vertiPrep, picType = "vertical")
    
horiPrep, cnts = imagePrep(imageHori, picType="horizontal")   
    
for a in np.arange(0,50):
    sweepline(cnts[a], horiPrep, picType = "horizontal")   
    
added = cv2.add(blackCanvasVerti, blackCanvasHori)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 50,50)
cv2.imshow('image',blackCanvasVerti)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 50,50)
cv2.imshow('image',blackCanvasHori)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 50,50)
cv2.imshow('image',added)
cv2.waitKey(0)
cv2.destroyAllWindows()
