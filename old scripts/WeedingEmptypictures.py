# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:24:50 2017

@author: cmd
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

gooddir   = 'Z:\\faellesmappe\\cmd\\MartinKarlsson\\tiny_pics\\good\\'
baddir   = 'Z:\\faellesmappe\\cmd\\MartinKarlsson\\tiny_pics\\bad\\'
gooddirsmall   = 'Z:\\faellesmappe\\cmd\\MartinKarlsson\\tiny_pics\\goodsmall\\'
baddirsmall   = 'Z:\\faellesmappe\\cmd\\MartinKarlsson\\tiny_pics\\badsmall\\'
savedir   = 'Z:\\faellesmappe\\cmd\\MartinKarlsson\\tiny_pics\\'
minipics_np = np.load(savedir+"allplots.npy")
minipicssmall_np = np.load(savedir+"allplotssmall.npy")

sumpix=[]
for pics in minipics_np:
    minipics_np_rev = 255-pics[30:45,5:45]
    minipics_np_rev = (minipics_np_rev>0)
    sumpix.append(np.sum(minipics_np_rev))


plt.hist(sumpix,bins=100)
plt.show()

good=[]
bad =[]
goodsmall=[]
badsmall =[]
idx = 0
for i in sumpix:
    if i>100:
        good.append(minipics_np[idx])
        goodsmall.append(minipicssmall_np[idx])
    else:    
        bad.append(minipics_np[idx])
        badsmall.append(minipicssmall_np[idx])
    idx +=1

np.save(gooddir+"allplotsgood", good)    
np.save(baddir+"allplotsbad", bad)    
np.save(gooddirsmall+"allplotsgoodsmall", goodsmall)    
np.save(baddirsmall+"allplotsbadsmall", badsmall)    

idx = 0
for pics in good:
    cv2.imwrite(gooddir+"tiny_good"+str(idx)+".JPG",pics)
    idx +=1

idx = 0
for pics in bad:
    cv2.imwrite(baddir+"tiny_bad"+str(idx)+".JPG",pics)
    idx +=1
    
idx = 0
for pics in goodsmall:
    cv2.imwrite(gooddirsmall+"tinysmall_good"+str(idx)+".JPG",pics)
    idx +=1

idx = 0
for pics in badsmall:
    cv2.imwrite(baddirsmall+"tinysmall_bad"+str(idx)+".JPG",pics)
    idx +=1
    