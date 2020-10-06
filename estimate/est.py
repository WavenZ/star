import cv2
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

from PIL import Image
from cv2 import cv2
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import os

def pca(points, xref, yref):
    '''incipal component analysis (PCA) is used to estimate 
    the motion direction of star points.
        
    Args：
        points: Star coordinates，shape = (2, n)
        xref: The x-coordinate of the upper left corner of the window.
        yref: The y-coordinate of the upper right corner of the window.
    
    Returns:
        The slope (k) and intercept (b) of the linear equation.
    '''
    # Transpose.
    data = points.T
    
    # PCA
    pca = PCA()
    pca.fit(data)
    if pca.components_[0, 0]:   # Avoid division by zero error.
        k = pca.components_[0, 1] / pca.components_[0, 0]
    else:
        k = 999999
    if k * (xref + np.mean(points[0])):     # Avoid division by zero error.
        b = np.mean(yref + points[1]) + 1 / k * (xref + np.mean(points[0]))
    else:
        b = 999999
    
    # dx and dy are the x and y ranges，respectively.
    # Used to set the limit of the first principal component（lower limit）.
    dx = np.max(data[:, 0]) - np.min(data[:, 0])
    dy = np.max(data[:, 1]) - np.min(data[:, 1])
    # print(dx, dy, end = ' ')
    # print(pca.explained_variance_ratio_[0])
    
    # limit: Minimun value of first principal compnent，which is in the 
    # range of [0.9 - 0.99]
    # The shorter the star is, the smaller the requirement for 
    # the size of the first principal component is.
    limit = 0.9 + 0.003 * (dx + dy)
    if limit > 0.99:
        limit = 0.99
    # The first principal component is greater than the limit.
    if pca.explained_variance_ratio_[0] > limit: 
        # print(pca.explained_variance_ratio_[0])
        return k, pca.explained_variance_ratio_[0]
    return 0, 0 # Directly identified as a fake star.

def threshold(img, n):
    """Thresholding.

    Search the first N gray scales in the window according to 
    the gray distribution histogram.
    
    Args: 
        img: Target mage.
        n: The first N gray scale that will be threshold to 255.

    Returns:
        Image after thresholding.
    """
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    th = 0
    for k in range(256):
        if hist[255 - k] != 0:
            n = n - 1
            if(n == 0):
                th = 255 - k - 1
                break
    # print(th)
    _, thImg = cv2.threshold(img,th,255,cv2.THRESH_BINARY)
    return thImg

def Cluster(coordins):
    '''Used to filter out noise far away from stars.

    Args:
        coordins: shape = (2, n)
    
    Returns: 
        Results of clustering (several groups).
    '''
    data = coordins.T
    # eps: The max distance between classes.
    # min_samples: The min number of points in a class.
    pred = DBSCAN(eps = 20, min_samples = 5).fit_predict(data)
    groups = []
    for i in np.unique(pred):
        cnt = list(pred).count(i)
        # The number of points in a single class is grater than 90% of the total.
        if cnt > 0.9 * pred.size: 
            groups.append(coordins[:, np.where(pred == i)].squeeze(1))
    return groups

def mean_value(values):
    '''Calculate the mean value by nearest 80% values。
    
    Args: 
        values: The set that we need to calculate the mean value.
    
    Returns:
        Mean value.
    '''
    if len(values) < 2:
        return np.mean(values)
    # Substract the mean value.
    diff = np.abs(values - np.mean(values))
    # Sorted by error.
    data = np.vstack((diff, values)).T
    data = data[np.argsort(data[:,0])]
    return np.mean(data[:round(len(data) * 0.8), 1])

def mean_value2(values, limit):
    '''Calculate the mean value.

    Filter out some invalid data by limiting the variance.

    Args:
        values: The set that we need to calculate the mean value.
        limit: The limit of variance.
    
    Returns:
        Mean value.
    '''

    # Directly return mean value while num of values less than 2.
    n = len(values)
    if n <= 2:
        return np.mean(values)
    
    # Sort first.
    values = np.sort(values)
    std = np.std(values)

    # Iterative calculation.
    while len(values) >= max(2, 0.8 * n) or std > limit:
        stda = np.std(values[1:])
        stdb = np.std(values[: -1])
        if stda < stdb:
            std = stda
            values = values[1:]
        else:
            std = stdb
            values = values[: -1]
    return np.mean(values)

def Direction_estimate(image):
    '''Estimate the direction of a single image.'''
    
    # Size of window and image.
    winsize = 100 
    rowsize, colsize = image.shape
    
    # Result.
    Theta, Linear = list([]), list([])

    for i in range(rowsize // winsize):
        for j in range(colsize // winsize):
            # A window of star image.
            window = image[i * winsize : (i + 1) * winsize, j * winsize : (j + 1) * winsize]

            mean = np.mean(window)
            # Skip while it's too bright.
            if mean > 180: 
                continue
            
            # Threshoulding.
            thImg = threshold(window, 6) 
            
            # Get the coordinates of positive points.
            points = np.array(np.where(thImg == 255))

            # Skip while the distribution is too scattered.
            if np.std(points[0]) + np.std(points[1]) > 50: 
                continue

            # Adjust the window so that the stars are in the middle of the window.
            dx, dy = np.mean(points, 1).astype(np.int32) - 50
            if i * winsize + dx < 0 or (i + 1) * winsize + dx >= 2048:
                dx = 0
            if j * winsize + dy < 0 or (j + 1) * winsize + dy >= 2048:
                dy = 0
            if dx or dy:
                window = image[i * winsize + dx: (i + 1) * winsize + dx,
                            j * winsize + dy: (j + 1) * winsize + dy]
                # Threshoulding.
                thImg = threshold(window, 4) 
                
                # Get the coordinates of positive points.
                points = np.array(np.where(thImg == 255))

            # Coordinate transformation.
            coordins = np.vstack((points[1], winsize - 1 - points[0])) 
            
            # Filter out the noise and seperate different star in the window.
            groups = Cluster(coordins) 


            for group in groups:
                theta, linear = pca(group, winsize * j, image.shape[0] - winsize * (i + 1))
                # (0, 0) indicates that there are no stars in the window.
                if theta != 0:
                    Theta.append(theta)
                    Linear.append(linear)
                # print(len(list(points.T)), np.arctan(theta) * 180 / np.pi, linear)
                # plt.figure()
                # plt.imshow(np.hstack((thImg, window)), cmap = 'gray')
                # plt.show()

    # print(Theta, Linear)
    temp = np.vstack((Theta, Linear)).T
    temp = np.array(sorted(temp, key=lambda x: x[1]))
    # print(temp.shape)
    # print(np.arctan(temp) * 180 / np.pi)

    Theta = temp[len(Theta) // 3: , 0]

    # print(np.arctan(np.array(Theta)) * 180 / np.pi)
    
    # Remove invalid value.
    for i in range(len(Theta)):
        if Theta[i] == float('inf'):
            Theta[i] = -99999


    # Return if Theta is null.
    if len(Theta) == 0: 
        return 0

    # If it is greater than 10, it means that the motion direction of 
    # the star point is about ±90, and the mean value requires special treatment.
    if np.mean(np.abs(Theta)) > 10: 
        res = np.arctan(Theta) * 180 / np.pi
        res = list(map(lambda x : x if x > 0 else 180 + x, res))
        res = mean_value2(res, 5)
        if res > 90:
            res = res - 180
    else:
        res = np.arctan(mean_value2(np.array(Theta), 2)) * 180 / np.pi
    # Result.
    return res 

def imshow(*images):
    '''Show the image.'''

    for image in images:
        plt.figure()
        plt.imshow(image, cmap='gray')
    plt.show()

def Kernel(size, width, theta):
    """Construct the convolution kernel.
        
    Args: size：size of kernel, (Height, Width)
          width：Width of the positive region.
          theta：Rotation angle.
    
    Notes：
        (size - width) shoule be even, so that the converlution kernel is symmetric.
    """

    temp = np.zeros((size + 10,size + 10))
    temp [(size - width) // 2 + 5:(size - width) // 2 + 5 + width,:] = 1
    temp = Image.fromarray(temp)
    temp = temp.rotate(theta)
    temp = np.array(temp)
    kernel = temp[5:-5, 5:-5]
    cnt = np.sum(kernel)
    for i in range(size):
        for j in range(size):
            kernel[i, j] = cnt / (cnt - size * size) if kernel[i, j] == 0 else kernel[i, j]
    kernel = kernel / 4
    return kernel


if __name__ == '__main__':
    file_path = r'C:\Users\14355\star\image\a'
    images = os.listdir(file_path)
    for image in images:
        if image[-3:] != 'png':
            continue
        print(image, end = ' ')
        src = cv2.imread(file_path + '\\' + image, 0)
        blured = cv2.blur(src, (3, 3))
        # blured = src
        direction =  Direction_estimate(blured)
        print(direction)