'''

==================================================
 Info:

 Written by: Trym Nygård 
 Last updated: March 2022

 Required python script "func.py"
'''

''' 
# PARAMETERS FOR HORSE IMAGE
M = 60       # Width of window
N = 60      # Height of window
X_L = 50    # X center position of the left window
X_R = 50    # X center position of the right window
Y_L = 30    # Y center position of the left window
Y_R = 30    # Y center position of the right window
''' 

# PARAMETERS FOR SEOUL IMAGE
M = 60       # Width of window (fraction cycles 67)
N = 60     # Height of window  (fraction cycles 67)
X_L = 150    # X center position of the left window
X_R = 150    # X center position of the right window
Y_L = 300    # Y center position of the left window
Y_R = 300    # Y center position of the right window

'''
# PARAMETERS FOR SALMON IMAGE
M = 30       # Width of window
N = 30       # Height of window
X_L = 1002    # X center position of the left window
Y_L = 315   # Y center position of the left window
X_R = 852    # X center position of the left window
Y_R = 315    # Y center position of the left window
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import func as fc
import time


showPlot = False

# Load image
imgL = cv.imread('seoul.jpg',0) 

SHIFTS = np.array([1,15,25,33,48,54,66,79])
SHIFTS2 = np.array([0.1,1.5,2.5,3.3,4.8,5.4,6.6,7.9])
def MAE(xc,xHat):
  return round(np.mean(np.abs(xHat-xc)),4)

def RMSE(xc,xHat):
  return round(np.sqrt(np.mean((xHat-xc)**2)),4)


SID = []
PC = []
QUAD_EST = []
GCC = []
GCC_SOBEL = []  
with open("result.txt", "w") as c:
    for i in range(len(SHIFTS)):
        # Create shifted image
        X_SHIFT = SHIFTS[i] # (fraction cycles 47)
        Y_SHIFT = 0
        translationMatrix = np.float32([[1, 0, X_SHIFT], [0, 1, Y_SHIFT]])
        imgR = cv.warpAffine(imgL, translationMatrix, (imgL.shape[1],imgL.shape[0]))

        # Resize
        SCALE_PERCENT = 10 # percent of original size
        resizedImgL, resizedimgR = fc.stereoImgResize(imgL,imgR,SCALE_PERCENT,cv.INTER_AREA)
        #print("Resized Image size: ", imgL.shape)

        print("*********************X-SHIFT = ",X_SHIFT/10,"]*********************")

        #Create window of features
        croppedL = fc.stereoImgCrop(resizedImgL, X_L, Y_L, M, N)
        croppedR = fc.stereoImgCrop(resizedimgR, X_R, Y_R, M, N)

        f, axarr = plt.subplots(1,2,figsize=(15,15))
        axarr[0].imshow(croppedL,cmap='gray')
        axarr[0].set_title("Left",fontsize=20)
        axarr[1].imshow(croppedR,cmap='gray')
        axarr[1].set_title("Right",fontsize=20)
        plt.show()

        #Apply window function
        winL = fc.windowFunc("blackman",croppedL) #funcType: blackman, hanning, 
        winR = fc.windowFunc("blackman",croppedR) #funcType: blackman, hanning

        #winL, winR, newX = fc.blockMatching(imgL,imgR, X_L, Y_L, M, N)      

        # Discrete Fourier Transform of both images 
        dftL = np.fft.fft2(winL)
        dftR = np.fft.fft2(winR)

        ''' 
        # Compute magnitude plot
        dftShiftL = np.fft.fftshift(dftL)
        dftShiftR = np.fft.fftshift(dftR)
        magSpecL = 20*np.log(np.abs(dftShiftL))
        magSpecR = 20*np.log(np.abs(dftShiftR))
        '''

        ''' 


        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(croppedL,cmap='gray')
        axarr[0,0].set_title("Reference image")
        axarr[0,1].imshow(croppedR,cmap='gray')
        axarr[0,1].set_title("Shifted image")
        axarr[1,0].imshow(winL,cmap='gray')
        axarr[1,1].imshow(winR,cmap='gray')
        plt.show()
        '''

        #**IMAGE REGISTRATION IN THE SPATIAL DOMAIN USING POC (INTEGER PRECISION)**
        #tic = time.perf_counter()
        #intShiftX =  fc.computePC(dftL,dftR,subpx = False, plot = showPlot)
        #toc = time.perf_counter()
        #print("intShiftX: ",intShiftX, "                  [INTEGER (Phase correlation)]","          Time: ", round(toc - tic,5),"s")

        #******IMAGE REGISTRATION IN THE FOURIER DOMAIN [SUB-PIXEL PRECISION]******
        subShiftX1 = fc.computePC(dftL,dftR, subpx=True, plot = showPlot)
        print("subShiftX: ",round(subShiftX1,3), "                [Phase Correlation]")

        subShiftX2 = fc.computeCCinter2(croppedL,croppedR, plot=showPlot)
        print("subShiftX: ",round(subShiftX2,3), "                [Cross Correlation 2 (Quadratic estimator)]")

        subShiftX3 = fc.computeGradCorr(croppedL,croppedR,gradMethod="hvdiff", plot=showPlot)
        print("subShiftX: ",round(subShiftX3,3), "                [Gradient Cross-Correlation (Cdiff)]")

        subShiftX4 = fc.computeGradCorr(croppedL,croppedR,gradMethod="sobel", plot=showPlot)
        print("subShiftX: ",round(subShiftX4,3), "                [Gradient Cross-Correlation (sobel)]")

        subShiftX5,_ = fc.computeSID(dftL,dftR,radius=0.4, magThres=0, plot = showPlot)
        print("subShiftX: ",round(subShiftX5, 3), "                [Subspace identification]")

 
        
        PC.append(round(subShiftX1,3))
        QUAD_EST.append(round(subShiftX2,3))
        GCC.append(round(subShiftX3,3))
        GCC_SOBEL.append(round(subShiftX4,3))
        SID.append(round(subShiftX5,3))
        
        positionStr = str(round(X_SHIFT/10,3)) +'&' + str(round(subShiftX1,3)) +'&'+str(round(subShiftX2,3)) +'&'+ str(round(subShiftX3,3)) +'&'+ str(round(subShiftX4,3)) +'&'+ str(round(subShiftX5,3))
        print(positionStr, file=c, flush=True)
        MAE(SHIFTS,subShiftX1)
    error = str(MAE(SHIFTS/10,PC)) +'&'+str( MAE(SHIFTS/10,QUAD_EST)) +'&'+ str( MAE(SHIFTS/10,GCC)) +'&'+ str( MAE(SHIFTS/10,GCC_SOBEL)) +'&'+ str( MAE(SHIFTS/10,SID))
    print(error, file=c, flush=True)
    error2 = str(RMSE(SHIFTS/10,PC)) +'&'+str( RMSE(SHIFTS/10,QUAD_EST)) +'&'+ str( RMSE(SHIFTS/10,GCC)) +'&'+ str( RMSE(SHIFTS/10,GCC_SOBEL)) +'&'+ str( RMSE(SHIFTS/10,SID))
    print(error2, file=c, flush=True)