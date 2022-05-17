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
M = 150       # Width of window (fraction cycles 67)
N = 150     # Height of window  (fraction cycles 67)
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
#imgL = cv.imread('horse.png',0) 
imgL = cv.imread('seoul.jpg',0) 
#imgL = cv.imread('images/stereo_left/L1.jpg',0)
#imgR = cv.imread('images/stereo_right/R1.jpg',0)

# Create shifted image
X_SHIFT = 64 # (fraction cycles 47)
Y_SHIFT = 0
translationMatrix = np.float32([[1, 0, X_SHIFT], [0, 1, Y_SHIFT]])
imgR = cv.warpAffine(imgL, translationMatrix, (imgL.shape[1],imgL.shape[0]))

# Resize
SCALE_PERCENT = 10 # percent of original size
imgL, imgR = fc.stereoImgResize(imgL,imgR,SCALE_PERCENT,cv.INTER_AREA)
#print("Resized Image size: ", imgL.shape)

print("*********************X-SHIFT = ",X_SHIFT/10,"]*********************")

#Create window of features
croppedL = fc.stereoImgCrop(imgL, X_L, Y_L, M, N)
croppedR = fc.stereoImgCrop(imgR, X_R, Y_R, M, N)

#Apply window function
winL = fc.windowFunc("hanning",croppedL) #funcType: blackman, hanning, 
winR = fc.windowFunc("hanning",croppedR) #funcType: blackman, hanning

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

''' '''


f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(croppedL,cmap='gray')
axarr[0,0].set_title("Reference image")
axarr[0,1].imshow(croppedR,cmap='gray')
axarr[0,1].set_title("Shifted image")
axarr[1,0].imshow(winL,cmap='gray')
axarr[1,1].imshow(winR,cmap='gray')
plt.show()

def computeTimePer(time,pOf):
    P = (abs(pOf - time)/pOf)*1
    return P

#**IMAGE REGISTRATION IN THE SPATIAL DOMAIN USING POC (INTEGER PRECISION)**
#tic = time.perf_counter()
#intShiftX =  fc.computePC(dftL,dftR,subpx = False, plot = showPlot)
#toc = time.perf_counter()
#print("intShiftX: ",intShiftX, "                  [INTEGER (Phase correlation)]","          Time: ", round(toc - tic,5),"s")

#******IMAGE REGISTRATION IN THE FOURIER DOMAIN [SUB-PIXEL PRECISION]******
tic = time.perf_counter()
subShiftX = fc.computeGradCorr(croppedL,croppedR,gradMethod="hvdiff", plot=showPlot)
toc = time.perf_counter()
print("subShiftX: ",round(subShiftX,3), "                [Gradient Cross-Correlation (Cdiff)]","           Time: ", round(computeTimePer(toc - tic,0.07359),1),"")

#tic = time.perf_counter()
#subShiftX = fc.computeGradCorr(croppedL,croppedR,gradMethod="scharr", plot=showPlot)
#toc = time.perf_counter()
#print("subShiftX: ",round(subShiftX,3), "                [Gradient Cross-Correlation (scharr)]","          Time: ", round(toc - tic,5),"s")

tic = time.perf_counter()
subShiftX = fc.computeGradCorr(croppedL,croppedR,gradMethod="sobel", plot=showPlot)
toc = time.perf_counter()
print("subShiftX: ",round(subShiftX,3), "                [Gradient Cross-Correlation (sobel)]","           Time: ", round(computeTimePer(toc - tic,0.07359),3),"")

tic = time.perf_counter()
subShiftX = fc.computeCCinter(croppedL,croppedR, plot=showPlot)
toc = time.perf_counter()
print("subShiftX: ",round(subShiftX,3), "                [Cross Correlation (2D spline)]","             Time: ", round(computeTimePer(toc - tic,0.07359),3),"")

tic = time.perf_counter()
subShiftX = fc.computeCCinter2(croppedL,croppedR, plot=showPlot)
toc = time.perf_counter()
print("subShiftX: ",round(subShiftX,3), "                [Cross Correlation 2 (Quadratic estimator)]","             Time: ", round(computeTimePer(toc - tic,0.07359),3),"")

tic = time.perf_counter()
subShiftX,subShiftY = fc.computeSID(dftL,dftR,radius=0.4, magThres=0, plot = showPlot)
toc = time.perf_counter()
print("subShiftX: ",round(subShiftX, 3), "                [Subspace identification]","                     Time: ", round(computeTimePer(toc - tic,0.07359),3),"")


tic = time.perf_counter()
subShiftX = fc.computePC(dftL,dftR, subpx=True, plot = showPlot)
toc = time.perf_counter()
print("subShiftX: ",round(subShiftX,3), "                [Phase Correlation]","                            Time: ", round(computeTimePer(toc - tic,0.07359),3),"%")

