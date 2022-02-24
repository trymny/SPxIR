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

def stereoImgCrop(imgL, imgR, xL,yL,xR,yR,N,M):
    winL = imgL[int(yL-N/2):int(yL+N/2), int(xL-M/2):int(xL+M/2)]
    winR = imgR[int(yR-N/2):int(yR+N/2), int(xR-M/2):int(xR+M/2)]
    print("Window size: ", winL.shape)
    return winL, winR

def stereoImgAddNoise(img):
    noise = np.random.uniform(img[:,:].min(),img[:,:].max(),len(img))
    return img + noise

def windowFunc(funcType, winL,winR):
    width,height = winL.shape
    Ww=eval('np.'+funcType+'(width)')
    Wh=eval('np.'+funcType+'(height)')
    windowFilter=np.outer(Ww,Wh)
    winL = winL * windowFilter
    winR = winR * windowFilter
    return winL,winR

def computeSubPixelShift(dominantSingularVec):
    singVecAngle = np.angle(dominantSingularVec)  #Find the angle
    unwrapSingVec = np.unwrap(singVecAngle)

    A = []
    for i in range(len(dominantSingularVec)):
        A.append(i)

    A = np.vstack([A, np.ones(len(A))]).T #Refered to as R in the paper
    fittedLine, res,rank,s = np.linalg.lstsq(A,unwrapSingVec,rcond=-1) #Equation 6 inv(R^TR)R^Tunwrap(v)

    mu = fittedLine[0] #slope of the fitted line
    #c = fittedLine[1] #abscissa of the fitted line

    return mu

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