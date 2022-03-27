'''
==================================================
 Additonal functions for phaseCorrelation.py
==================================================
 Info:

 Written by: Trym Nygård 
 Last updated: March 2022
 
'''

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2 as cv
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
            #dft[:,i] = 0.00001 # zeroes out column i
    return dft

def radMask(dft,param=0.6):
    N = dft.shape[1]
    radius = param*((N)/(2))
    maskedDft = dft[int(dft.shape[0]/2-radius):int(dft.shape[0]/2+radius),int(dft.shape[0]/2-radius):int(dft.shape[0]/2+radius)]
    return maskedDft

def computeSubSpaceID(dftL,dftR, radius=0.6, magThres=0, plot=False):
    
    M = dftL.shape[0]
    N = dftL.shape[1]

    dftShiftL = np.fft.fftshift(dftL)
    dftShiftR = np.fft.fftshift(dftR)
    
    #Mask out spectral components that lie outside a radius from the central peak
    maskedDftL = radMask(dftShiftL,radius)
    maskedDftR = radMask(dftShiftR,radius)
    
    #Mask out spectral components for DFT's that have peaks with magnitudes less than a threshold value
    #maskedDftL = magMask(maskedDftL,magThres)
    #maskedDftR = magMask(maskedDftR,magThres)

    # Normalized cross-power spectrum 
    R = (maskedDftL*np.conjugate(maskedDftR))/(np.abs(maskedDftL*np.conjugate(maskedDftR)))  

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
    cX = fittedLineV[1] #abscissa of the fitted line
    subShiftX = muX * (M / (2*np.pi)) #translational shift

    angleU = np.angle(dominantU)  #Find the angle
    unwrapU = np.unwrap(angleU)
    fittedLineU, res,rank,s = np.linalg.lstsq(A,unwrapU,rcond=-1) #Equation 6 inv(R^TR)R^Tunwrap(v)
    muY = fittedLineU[0] #slope of the fitted line
    #cY = fittedLineU[1] #abscissa of the fitted line
    subShiftY = muY * (N / (2*np.pi)) #translational shift

    if plot: 
        plotDFT("Frequency Spectrum Left Image",maskedDftL)
        plotDFT("Frequency Spectrum Right Image",maskedDftR)

        temp = unwrapV.shape[0]
        fx= (np.linspace(-temp/2,temp/2-1,temp)/temp);    # FM = number of pixel in x direction 
        fx = fx *2 * np.pi                    # Multiply with 2pi to get the (spatial) frequency 
        fx = np.reshape(fx, (-1, 1))          # Go from shape (M,) to (M,1)

        fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        axarr[0].set_title("Phase difference")
        axarr[0].grid(True)
        axarr[0].plot(fx, angleV)
        
        x = np.linspace(0,temp,temp)
        axarr[1].set_title("Unwrapped Phase difference")
        axarr[1].grid(True)
        axarr[1].plot(x, unwrapV)
        axarr[1].plot(x, muX*x+cX)
        axarr[1].text(0,max(unwrapV),"X-shift: "+str(round(subShiftX,3)),fontsize=12)
        axarr[1].text(max(x)/2,(max(unwrapV)+min(unwrapV))/2,str(muY))
        plt.show()
    
    
    return subShiftX,subShiftY

def computeGradCorr(img, img2, polyDeg = 2, nPoints = 200, plot = False):
    M = img.shape[0]

    gHor = np.zeros_like(img)
    gVer = np.zeros_like(img)
    gHor2 = np.zeros_like(img2)
    gVer2 = np.zeros_like(img2)

    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            gVer[i,j] = (int(img[i+1,j])-int(img[i-1,j]))
            gHor[i,j] = (int(img[i,j+1])-int(img[i,j-1]))
            gVer2[i,j] = (int(img2[i+1,j])-int(img2[i-1,j]))
            gHor2[i,j] = (int(img2[i,j+1])-int(img2[i,j-1]))
    
    g = gHor - np.imag(gVer)
    g2 = gHor2 - np.imag(gVer2)

    dftG = (np.fft.fft2(g))
    dftG2 = (np.fft.fft2(g2))
    
    cc = np.fft.ifft2(dftG*np.conjugate(dftG2))
     
    for i in range(int(M/2)):
        cc = np.roll(cc, -1, axis=1)

    cc = cc.real
    #plotSurface(cc)
    
    idxMax = np.unravel_index(cc.argmax(), cc.shape) 
    
    dist = 1
    y = np.array([cc[idxMax[0],idxMax[1]-dist], cc[idxMax[0],idxMax[1]], cc[idxMax[0],idxMax[1]+dist]])
    x = np.linspace(idxMax[1]-dist,idxMax[1]+dist,3).astype(int)

    fit = np.polyfit(x,y,polyDeg)
    p = np.poly1d(fit) 
    newX = np.linspace(idxMax[1]-dist,idxMax[1]+dist,nPoints)
    
    maxPeak = [newX[np.argmax(p(newX))],p(newX[np.argmax(p(newX))])]
    x_shift = int(M/2)-maxPeak[0]

    if plot:
        temp = M
        xx = np.linspace(0,temp,temp).astype(int)  
        fig, axarr = plt.subplots(1,1)
        axarr.grid(True)
        axarr.plot(xx,cc[idxMax[0],:] )
        axarr.plot(newX,p(newX))
        axarr.plot(x[0],y[0], marker="o", color="green")
        axarr.plot(x[1],y[1], marker="o", color="green")
        axarr.plot(x[2],y[2], marker="o", color="green")
        axarr.text(maxPeak[0],maxPeak[1],str(x_shift),horizontalalignment='right')
        plt.show()
    
    return x_shift

def computePOC(dftL,dftR, plot = False):

    # Normalized cross-power spectrum 
    R = (dftL*np.conjugate(dftR))/(np.abs(dftL*np.conjugate(dftR)))  

    # Taking the inverse discrete Fourier transform to find the phase correlation
    POC = np.fft.ifft2(R)

    # Using the fourier shif theorem to move the maximum peak
    POC = np.fft.fftshift(POC)
    
    # Compute magnitude
    POC = abs(POC)

    # Using a 2d Gaussian distribution to smooth the phase correlation and to remove false peaks
    POC = cv.GaussianBlur(POC,(5,5),0)

    if plot:
        print("POC Max Peak: ", np.max(POC))
        plt.figure()
        plt.imshow(POC,cmap='gray')
        plt.show()

    return POC

def computePOCshift(POC):
    # Determine the location of the max peak in the POC function
    idxPeak = np.unravel_index(POC.argmax(), POC.shape) # Extract indices of max peak from POC function 
    intShiftY = (POC.shape[0]/2-idxPeak[0])       # Compute integer shift Y from the max peak in POC function
    intShiftX = (POC.shape[1]/2-idxPeak[1])       # Compute integer shift X from the max peak in POC function
    return intShiftX,intShiftY

def blockMatching(imgL, imgR,X_L, Y_L, M, N, method="TM_CCORR_NORMED", winFunc = "blackman"):

    # Creating a window around the feature
    winL = stereoImgCrop(imgL, X_L, Y_L, M, N)

    # Applying blackman window 
    if winFunc != "False":
        winL = windowFunc(winFunc,winL) #funcType: blackman, hanning,

    x = int(M/2)
    maxVal = []
    minVal = []
    xList = []
    while(x < X_L):
        winR = stereoImgCrop(imgR, x, Y_L, M, N) 
        
        if winFunc != "False":
            winR = windowFunc(winFunc,winR)

        if method == "TM_CCORR_NORMED":
            res = cv.matchTemplate(winR.astype(np.float32),winL.astype(np.float32),cv.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            maxVal.append(max_val)
        elif method == "TM_CCOEFF_NORMED":
            res = cv.matchTemplate(winR.astype(np.float32),winL.astype(np.float32),cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            maxVal.append(max_val)
        elif method == "TM_SQDIFF_NORMED":
            res = cv.matchTemplate(winR.astype(np.float32),winL.astype(np.float32),cv.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            minVal.append(min_val)
        elif method == "POC":
            dftL = np.fft.fft2(winL)
            dftR = np.fft.fft2(winR)
            POC = computePOC(dftL,dftR)
            maxVal.append(np.max(POC))
        xList.append(x)
        x=x+1
    
    if winFunc != "False":
        winR = windowFunc(winFunc,winR)
    
    if method == "TM_SQDIFF_NORMED":
        #print("minVal: ", minVal[np.argmin(minVal)], " x: ", xList[np.argmin(minVal)])
        xShift = xList[np.argmin(minVal)]
    else:
        #print("maxVal: ", maxVal[np.argmax(maxVal)], " x: ", xList[np.argmax(maxVal)])
        xShift = xList[np.argmax(maxVal)]

    winR = stereoImgCrop(imgR, xShift, Y_L, M, N) 

    return winL, winR, xShift

def plotPhaseDifference(dftL,dftR,radius=0.6,rStepSize=2, cStepSize=2, aa=True):

    dftShiftL = np.fft.fftshift(dftL)
    dftShiftR = np.fft.fftshift(dftR)

    #Mask out spectral components that lie outside a radius from the central peak
    maskedDftL = radMask(dftShiftL,radius)
    maskedDftR = radMask(dftShiftR,radius)
    
    #Mask out spectral components for DFT's that have peaks with magnitudes less than a threshold value
    #maskedDftL = magMask(maskedDftL,magThres)
    #maskedDftR = magMask(maskedDftR,0)

    R = (maskedDftL*np.conjugate(maskedDftR))/(np.abs(maskedDftL*np.conjugate(maskedDftR)))  
    
    theta = np.arctan2(R.imag,R.real)  #Can also use theta = np.angle(R) 

    # Plot of phase component before removal of integer shift (figure 3)
    fig = plt.figure()
    xData, yData = np.mgrid[(-theta.shape[0]/2):(theta.shape[0]/2), (-(theta.shape[1])/2):(theta.shape[1]/2)]
    ax = fig.gca(projection='3d')
    ax.set_title('Phase differences')
    ax.plot_wireframe(xData,yData,theta,rstride=rStepSize, cstride=cStepSize,antialiased=aa)
    plt.show()

def plotSurface(func, wireframe=False,rStepSize=1, cStepSize=1, aa=True):
    #*************************Phase correlation surface plot*********************************
    xx, yy = np.mgrid[(-func.shape[0]/2):(func.shape[0]/2), (-(func.shape[1])/2):(func.shape[1]/2)]

    #Add fancy color mapping
    minn, maxx = func.min(), func.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(func)

    #create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if not wireframe:
        ax.plot_surface(xx, yy, func , facecolors=fcolors,rstride=rStepSize, cstride=cStepSize,linewidth=0, antialiased=aa, shade=False)
    else:
        ax.plot_wireframe(xx, yy, func, rstride=rStepSize, cstride=cStepSize,antialiased=aa)

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