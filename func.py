'''
==================================================
 Additonal functions for phaseCorrelation.py
==================================================
 Info:

 Written by: Trym Nygård 
 Last updated: Februar 2022
 
'''

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2 as cv
from scipy import signal, misc
from mpl_toolkits.mplot3d import Axes3D

def stereoImgResize(imgL,imgR, scale_percent,interMethod):
    width = int(imgL.shape[1] * scale_percent / 100)
    height = int(imgL.shape[0] * scale_percent / 100)
    dimL = (width, height)
    imgL = cv.resize(imgL, dimL, interpolation = interMethod)
    width = int(imgR.shape[1] * scale_percent / 100)
    height = int(imgR.shape[0] * scale_percent / 100)
    dimR = (width, height)
    imgR = cv.resize(imgR, dimR, interpolation = interMethod)

    return imgL, imgR

def stereoImgCrop(img,X,Y,N,M):
    win = img[int(Y-(N/2)):int(Y+(N/2)), int(X-(M/2)):int(X+(M/2))]
    #print("Window size: ", winL.shape)
    return win

def stereoImgAddNoise(img):
    img = (img - np.mean(img)) / np.std(img)
    return img 

def windowFunc(funcType, win):
    width,height = win.shape
    Ww=eval('np.'+funcType+'(width)')
    Wh=eval('np.'+funcType+'(height)')
    windowFilter=np.outer(Ww,Wh)
    win = win * windowFilter
    return win

def magMask(dft,param=0):
    for i in range(dft.shape[0]):
        maxValue = abs(np.max(dft[i]))
        if(maxValue <= param):
            dft[i] = 0.00001   # zeroes out row i
            dft[:,i] = 0.00001 # zeroes out column i
    return dft

def radMask(dft,param=0.6):
    N = dft.shape[1]
    radius = param*((N)/(2))
    maskedDft = dft[int(dft.shape[0]/2-radius):int(dft.shape[0]/2+radius),int(dft.shape[0]/2-radius):int(dft.shape[0]/2+radius)]
    return maskedDft

def computeSubPixelShift(dftL,dftR, radius=0.6, magThres=0):
    
    M = dftL.shape[0]
    N = dftL.shape[1]

    dftShiftL = np.fft.fftshift(dftL)
    dftShiftR = np.fft.fftshift(dftR)
    
    #Mask out spectral components that lie outside a radius from the central peak
    maskedDftL = radMask(dftShiftL,radius)
    maskedDftR = radMask(dftShiftR,radius)
    
    #Mask out spectral components for DFT's that have peaks with magnitudes less than a threshold value
    #maskedDftL = magMask(maskedDftL,magThres)
    maskedDftR = magMask(maskedDftR,magThres)

    # Normalized cross-power spectrum 
    R = (maskedDftL*np.conjugate(maskedDftR))/(np.abs(maskedDftL*np.conjugate(maskedDftR)))  

    #plotDFT("Frequency Spectrum Left Image",maskedDftL)
    #plotDFT("Frequency Spectrum Right Image",maskedDftR)

    U,S,V  = np.linalg.svd(R) #Reduce from 2D to 1D 

    dominantV = V[np.argmax(S),:]
    dominantU = U[:,np.argmax(S)]

    A = []
    for i in range(len(V)):
        A.append(i)
    A = np.vstack([A, np.ones(len(A))]).T #Refered to as R in the paper

    angleV = np.angle(dominantV)  #Find the angle
    unwrapV = np.unwrap(angleV)
    fittedLineV, res,rank,s = np.linalg.lstsq(A,unwrapV,rcond=-1) #Equation 6 inv(R^TR)R^Tunwrap(v)
    muX = fittedLineV[0] #slope of the fitted line
    #cX = fittedLineV[1] #abscissa of the fitted line
    subShiftX = muX * (M / (2*np.pi)) #translational shift

    angleU = np.angle(dominantU)  #Find the angle
    unwrapU = np.unwrap(angleU)
    fittedLineU, res,rank,s = np.linalg.lstsq(A,unwrapU,rcond=-1) #Equation 6 inv(R^TR)R^Tunwrap(v)
    muY = fittedLineU[0] #slope of the fitted line
    #cY = fittedLineU[1] #abscissa of the fitted line
    subShiftY = muY * (N / (2*np.pi)) #translational shift

    return subShiftX,subShiftY

def computePOC(dftL,dftR):

    # Normalized cross-power spectrum 
    R = (dftL*np.conjugate(dftR))/(np.abs(dftL*np.conjugate(dftR)))  

    # Taking the inverse discrete Fourier transform to find the phase correlation (Equation 7)
    POC = np.fft.ifft2(R)

    # Using the fourier shif theorem to move the maximum peak
    POC = np.fft.fftshift(POC)
    
    # Compute magnitude
    POC = abs(POC)

    # Using a 2d Gaussian distribution to smooth the phase correlation and to remove false peaks
    POC = cv.GaussianBlur(POC,(5,5),0)

    return POC

def similarityMeasure(imgL, imgR,X_L, Y_L, M, N, method="TM_CCORR_NORMED", winFunc = "blackman"):

    # Creating a window around the feature
    winL = stereoImgCrop(imgL, X_L, Y_L, M, N)

    # Applying blackman window 
    winL = windowFunc("blackman",winL) #funcType: blackman, hanning,

    x = X_L-200
    maxPeak = []
    xList = []
    while(x < X_L+M):
        winR = stereoImgCrop(imgR, x, Y_L, M, N) 
        winR = windowFunc(winFunc,winR)

        if method == "TM_CCORR_NORMED":
            res = cv.matchTemplate(winR.astype(np.float32),winL.astype(np.float32),cv.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            maxPeak.append(max_val)
        elif method == "TM_CCOEFF_NORMED":
            res = cv.matchTemplate(winR.astype(np.float32),winL.astype(np.float32),cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            maxPeak.append(max_val)
        elif method == "POC":
            dftL = np.fft.fft2(winL)
            dftR = np.fft.fft2(winR)
            POC = computePOC(dftL,dftR)
            maxPeak.append(np.max(POC))
        
        xList.append(x)
        x=x+1
    #print("maxPeak: ", maxPeak[np.argmax(maxPeak)], " x: ", xList[np.argmax(maxPeak)])
    winR = stereoImgCrop(imgR, xList[np.argmax(maxPeak)], Y_L, M, N) 
    winR = windowFunc(winFunc,winR)
    return winL, winR,xList[np.argmax(maxPeak)]

def plotPhaseDifference(dftL,dftR,rStepSize=2, cStepSize=2, aa=True):

    dftShiftL = np.fft.fftshift(dftL)
    dftShiftR = np.fft.fftshift(dftR)

    # Normalized cross-power spectrum 
    R = (dftShiftL*np.conjugate(dftShiftR))/(np.abs(dftShiftL*np.conjugate(dftShiftR)))  
    
    theta = np.arctan2(R.imag,R.real)  #Can also use theta = np.angle(R) 

    # Plot of phase component before removal of integer shift (figure 3)
    fig = plt.figure()
    xData, yData = np.mgrid[(-theta.shape[0]/2):(theta.shape[0]/2), (-(theta.shape[1])/2):(theta.shape[1]/2)]
    ax = fig.gca(projection='3d')
    ax.set_title('Phase differences')
    ax.plot_wireframe(xData,yData,theta,rstride=rStepSize, cstride=cStepSize,antialiased=aa)
    plt.show()

def computePOCshift(POC):
    # Determine the location of the max peak in the POC function
    idxPeak = np.unravel_index(POC.argmax(), POC.shape) # Extract indices of max peak from POC function
    intShiftY = (POC.shape[0]/2-idxPeak[0])       # Compute integer shift Y from the max peak in POC function
    intShiftX = (POC.shape[1]/2-idxPeak[1])       # Compute integer shift X from the max peak in POC function
    return intShiftX,intShiftY

def plotPOCsurface(POC, wireframe=False,rStepSize=1, cStepSize=1, aa=True):
    #*************************Phase correlation surface plot*********************************
    xx, yy = np.mgrid[(-POC.shape[0]/2):(POC.shape[0]/2), (-(POC.shape[1])/2):(POC.shape[1]/2)]

    #Add fancy color mapping
    minn, maxx = POC.min(), POC.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(POC)

    #create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if not wireframe:
        ax.plot_surface(xx, yy, POC , facecolors=fcolors,rstride=rStepSize, cstride=cStepSize,linewidth=0, antialiased=aa, shade=False)
    else:
        ax.plot_wireframe(xx, yy, POC, rstride=rStepSize, cstride=cStepSize,antialiased=aa)

    plt.show()

def displayInfoPOC(POC):
    print("POC Max Peak: ", np.max(POC))

def plotPOC(POC):
    plt.figure()
    plt.imshow(POC,cmap='gray')
    plt.show()

def plotDFT(title, dft):
    M = dft.shape[0]
    fx= (np.linspace(-M/2,M/2-1,M)/M);    # FM = number of pixel in x direction 
    fx = fx *2 * np.pi                    # Multiply with 2pi to get the (spatial) frequency 
    fx = np.reshape(fx, (-1, 1))          # Go from shape (M,) to (M,1)
    fig = plt.figure()
    ax = fig.gca()
    ax.grid(True)
    ax.set_title(title)
    ax.plot(fx,abs(dft))
    plt.show()