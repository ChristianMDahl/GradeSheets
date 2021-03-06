#import sys
#import os
#script_dir = os.path.dirname(os.path.realpath('__file__'))
#rel_path   = '\\library'
# Add the directory containing your module to the Python path (wants absolute paths)
#sys.path.append(os.path.abspath(script_dir+rel_path))

from cut_out_tables import *
datadir   = 'X:\\martin_image_data\\Bjuv Stad\\Sheet2\\'    
#savedir   = 'Z:\\faellesmappe\\cmd\\MartinKarlsson\\tiny_pics\\'
imagesall =  getImagesInDirectory(datadir)
img2                 = cv2.imread(datadir + imagesall[0],0)
y1,x1,v1,h1          = cropping_GS_images(img2)
v1cleansheet         = horizontal_sweepline(v1)
h1cleansheet         = horizontal_sweepline(h1)
img2_cropped         = img2[y1[0]:y1[1],x1[0]:x1[1]]
ddd                  = cv2.add(v1cleansheet.T,h1cleansheet)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('image', ratio[1],ratio[0])
cv2.resizeWindow('image', 50,50)
cv2.imshow('image', v1cleansheet.T)
cv2.waitKey(0)
cv2.destroyAllWindows()

dd1, ctsk, hierarchy = cv2.findContours(ddd.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        areaI    = []
        wI       = [] 
        hI       = []
        idx      = 0
        ratio    = (28,28)  
        for c_ in ctsk:
            x,y,w,h         = cv2.boundingRect(c_)    
            area            = cv2.contourArea(c_)
            image           = img2_cropped.copy()[y:y+h,x:x+w]    
            if w>40 and w<60 and h>60 and h<80:
                try:
                    tresh            = threshold_otsu(image.copy())                          
                    __,image         = cv2.threshold(image.copy(),tresh,255,0)
                    wI.append(w)
                    hI.append(h)
                    areaI.append(area)
                    dsf_add          = cv2.resize(image,ratio,interpolation = cv2.INTER_AREA)
                    minipics.append(dsf_add)          
#                    cv2.imwrite(savedir+"tiny_small"+str(idx)+"_"+pics,dsf_add)
                    idx              = idx + 1
                except:
                    pass    
    except:
        pass

#Saving minipics     
minipics_np = np.asarray(minipics)
np.save(savedir+"allplotssmall", minipics_np)

## Classifying pics in good (sufficiently many informative pixels) and bad (too few informative pixels= blanc field in grading sheet)
sumpix=[]
for pics in minipics_np:
    minipics_np_rev = 255-pics[30:45,5:45]
    minipics_np_rev = (minipics_np_rev>0)
    sumpix.append(np.sum(minipics_np_rev))

good=[]
bad =[]    
idx = 0
for i in sumpix:
    if i>100:
        good.append(minipics_np[idx])
    else:    
        bad.append(minipics_np[idx])
    idx +=1
np.save("allplotsgood", good)    
np.save("allplotsbad", bad)   
