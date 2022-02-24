'''
==================================================
    Sub-pixel image registration based on subspace 
Identification extension of the Phase Correlation Method
==================================================
 Info:

 Written by: Trym Nygård 
 Last updated: Februar 2022

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
winL, winR = fc.stereoImgCrop(imgL, imgR, X_L, Y_L, X_R, Y_R, M, N)
'''
 
# PARAMETERS FOR SEOUL IMAGE
M = 50       # Width of window
N = 50       # Height of window
X_L = 150    # X center position of the left window
X_R = 150    # X center position of the right window
Y_L = 300    # Y center position of the left window
Y_R = 300    # Y center position of the right window

'''
# PARAMETERS FOR SALMON IMAGE
M = 40       # Width of window
N = 40       # Height of window
X_L = 1002    # X center position of the left window
X_R = 850   # X center position of the right window
Y_L = 315    # Y center position of the left window
Y_R = 315    # Y center position of the right window
'''

from tkinter import W
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import func as fc

from mpl_toolkits.mplot3d import Axes3D

# Load image
#imgL = cv.imread('horse.png',0) 
imgL = cv.imread('seoul.jpg',0) 
#imgL = cv.imread('images/stereo_left/L1.jpg',0)
#imgR = cv.imread('images/stereo_right/R1.jpg',0)

# Create shifted image
X_SHIFT = 31
Y_SHIFT = 0
translationMatrix = np.float32([[1, 0, X_SHIFT], [0, 1, Y_SHIFT]])
imgR = cv.warpAffine(imgL, translationMatrix, (imgL.shape[1],imgL.shape[0]))

# Resize
SCALE_PERCENT = 10 # percent of original size
imgL, imgR = fc.stereoImgResize(imgL,imgR,SCALE_PERCENT,cv.INTER_AREA)
print("Resized Image size: ", imgL.shape)

# Creating a window around the feature
winL, winR = fc.stereoImgCrop(imgL, imgR, X_L, Y_L, X_R, Y_R, M, N)

# Applying blackman window 
winL, winR = fc.windowFunc("blackman",winL,winR) #funcType: blackman, hanning, 

# Discrete Fourier Transform of both images (Equation 3 and 4)
dftL = np.fft.fft2(winL)
dftR = np.fft.fft2(winR)

# Compute magnitude plot
dftShiftL = np.fft.fftshift(dftL)
dftShiftR = np.fft.fftshift(dftR)
magSpecL = 20*np.log(np.abs(dftShiftL))
magSpecR = 20*np.log(np.abs(dftShiftR))

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(winL,cmap='gray')
axarr[0,1].imshow(winR,cmap='gray')
axarr[1,0].imshow(magSpecL,cmap='gray')
axarr[1,1].imshow(magSpecR,cmap='gray')
plt.show()

#**IMAGE REGISTRATION IN THE SPATIAL DOMAIN USING POC (INTEGER PRECISION)**

POC = fc.computePOC(dftL,dftR)
#fc.plotPOCsurface(POC, wireframe=True)
#fc.plotPOC(POC)
intShiftX,intShiftY =  fc.computePOCshift(POC)

print("intShiftX: ",intShiftX, " intShiftY: ",intShiftY, "       [INTEGER SHIFT FROM POC]")

#******IMAGE REGISTRATION IN THE FOURIER DOMAIN [SUB-PIXEL PRECISION]******
radius = int(0.4*((N)/(2)))
maskedDftL = dftShiftL[int(dftShiftL.shape[0]/2-radius):int(dftShiftL.shape[0]/2+radius),int(dftShiftL.shape[0]/2-radius):int(dftShiftL.shape[0]/2+radius)]
maskedDftR = dftShiftR[int(dftShiftR.shape[0]/2-radius):int(dftShiftR.shape[0]/2+radius),int(dftShiftR.shape[0]/2-radius):int(dftShiftR.shape[0]/2+radius)]

#fc.plotDFT("Frequency Spectrum Left Image",maskedDftL)
#fc.plotDFT("Frequency Spectrum Right Image",maskedDftR)

# Normalized cross-power spectrum 
R = (maskedDftL*np.conjugate(maskedDftR))/(np.abs(maskedDftL*np.conjugate(maskedDftR)))  

u,s,v  = np.linalg.svd(R) #Reduce from 2D to 1D 

dominantV = v[np.argmax(s),:]
muX = fc.computeSubPixelShift(dominantV)
subShiftX = muX * (M / (2*np.pi)) #translational shift

dominantU = u[:,np.argmax(s)]
muY = fc.computeSubPixelShift(dominantU)
subShiftY = muY * (M / (2*np.pi)) #translational shift

print("subShiftX: ",subShiftX, " subShiftY: ",subShiftY, "    [INTEGER SHIFT with SUBPIXEL SHIFT]")


#*****************************PHASE PLOT*******************************
fc.plotPhaseDifference(dftL,dftR)

