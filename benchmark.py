'''

==================================================
 Info:

 Written by: Trym Nygård 
 Last updated: March 2022

 Required python script "func.py"
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import func as fc
import time

def RMSE(xc,xHat):
  return round(np.sqrt(np.mean((xHat-xc)**2)),4)

def MSE(xc,xHat):
  return round(np.mean((xHat-xc)**2),4)

def MAE(xc,xHat):
  return round(np.mean(np.abs(xHat-xc)),4)

showPlot = False

# Load image
imgL = cv.imread('seoul.jpg',0) 


Nn = []
SIDe  = []
PCe = []
QUAD_ESTe = []
GCCe  = []
GCC_SOBELe  = []
for n in range (150,10,-10):
    Nn.append(n)
    # PARAMETERS FOR SEOUL IMAGE
    M = n       # Width of window (fraction cycles 67)
    N = n     # Height of window  (fraction cycles 67)
    X_L = 150    # X center position of the left window
    X_R = 150    # X center position of the right window
    Y_L = 300    # Y center position of the left window
    Y_R = 300    # Y center position of the right window
    GT = []
    SID = []
    PC = []
    QUAD_EST = []
    GCC = []
    GCC_SOBEL = []
    for i in range (0,40,5):

        # Create shifted image
        X_SHIFT = 1+i # (fraction cycles 47)
        Y_SHIFT = 0
        GT.append(X_SHIFT/10)
        translationMatrix = np.float32([[1, 0, X_SHIFT], [0, 1, Y_SHIFT]])
        imgR = cv.warpAffine(imgL, translationMatrix, (imgL.shape[1],imgL.shape[0]))

        # Resize
        SCALE_PERCENT = 10 # percent of original size
        resizedImgL, resizedimgR = fc.stereoImgResize(imgL,imgR,SCALE_PERCENT,cv.INTER_AREA)
        #print("Resized Image size: ", imgL.shape)

        #Create window of features
        croppedL = fc.stereoImgCrop(resizedImgL, X_L, Y_L, M, N)
        croppedR = fc.stereoImgCrop(resizedimgR, X_R, Y_R, M, N)

        #Apply window function
        winL = fc.windowFunc("blackman",croppedL) #funcType: blackman, hanning, 
        winR = fc.windowFunc("blackman",croppedR) #funcType: blackman, hanning

        # Discrete Fourier Transform of both images 
        dftL = np.fft.fft2(winL)
        dftR = np.fft.fft2(winR)

        subShiftX,_ = fc.computeSID(dftL,dftR,radius=0.4, magThres=0, plot = showPlot)
        SID.append(round(subShiftX,3))
        
        subShiftX = fc.computePC(dftL,dftR, subpx=True, plot = showPlot)
        PC.append(round(subShiftX,3))

        subShiftX = fc.computeCCinter2(croppedL,croppedR, plot=showPlot)
        QUAD_EST.append(round(subShiftX,3))

        subShiftX = fc.computeGradCorr(croppedL,croppedR,gradMethod="hvdiff", plot=showPlot)
        GCC.append(round(subShiftX,3))

        subShiftX = fc.computeGradCorr(croppedL,croppedR,gradMethod="sobel", plot=showPlot)
        GCC_SOBEL.append(round(subShiftX,3))
        

    print("dfdf")
    GTx = np.array(GT)
    SIDx = np.array(SID)
    PCx = np.array(PC)
    GCCx = np.array(GCC)
    GCC_SOBELx = np.array(GCC_SOBEL)
    QUAD_ESTx = np.array(QUAD_EST)
    print("**********************************",n,"********************************")

    print("PC:        ", MAE(GTx,PCx), RMSE(GTx,PCx))
    PCe.append(MAE(GTx,PCx))
    print("QUAD_EST:  ", MAE(GTx,QUAD_ESTx), RMSE(GTx,QUAD_ESTx))
    QUAD_ESTe.append(MAE(GTx,QUAD_ESTx))
    print("GCC:       ", MAE(GTx,GCCx), RMSE(GTx,GCCx))
    GCCe.append(MAE(GTx,GCCx))
    print("GCC SOBEL: ", MAE(GTx,GCC_SOBELx), RMSE(GTx,GCC_SOBELx),)
    GCC_SOBELe.append(MAE(GTx,GCC_SOBELx))
    print("SID:       ", MAE(GTx,SIDx), RMSE(GTx,SIDx))
    SIDe.append(MAE(GTx,SIDx))

print(SIDe)

fig = plt.figure(figsize=(16, 10))
ax = fig.gca()
ax.set_xlabel('Window size [Px]',fontsize=20)
ax.set_ylabel('Mean absolute error [Px]',fontsize=20)
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20

plt.gca().invert_xaxis()
ax.grid(True)
line1, = ax.plot(Nn,PCe)
line1.set_label('Phase correlation')
line2, = ax.plot(Nn,QUAD_ESTe)
line2.set_label('Quadratic estimator')
line3, = ax.plot(Nn,GCCe)
line3.set_label('Gradient cross-correlation')
line4, = ax.plot(Nn,GCC_SOBELe)
line4.set_label('Gradient cross-correlation (sobel)')
line5, = ax.plot(Nn,SIDe)
line5.set_label('Subspace identification')
ax.tick_params(labelsize=15)
ax.legend(fontsize=15)
plt.show()


