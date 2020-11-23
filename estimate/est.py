import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import numpy.ctypeslib as npct

from ctypes import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from cv2 import cv2
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import os

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def pca(points, xref, yref):
    '''incipal component analysis (PCA) is used to estimate 
    the motion direction of star points.

    Args:
        points: Star coordinates, shape = (2, n)
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
    if pca.components_[0, 1]:   # Avoid division by zero error.
        k = - pca.components_[0, 0] / pca.components_[0, 1]
    else:
        k = 999999
    if k * (xref + np.mean(points[0])):     # Avoid division by zero error.
        # b = np.mean(yref + points[1]) + 1 / k * (xref + np.mean(points[0]))
        b = yref + np.mean(points[1]) - k * (xref + np.mean(points[0]))
    else:
        b = 999999
    
    # print('center =', yref + np.mean(points[0]), xref + np.mean(points[1]))
    # print('k =', k, 'b =', b)

    # dx and dy are the x and y ranges, respectively.
    # Used to set the limit of the first principal component (lower limit).
    dx = np.max(data[:, 0]) - np.min(data[:, 0])
    dy = np.max(data[:, 1]) - np.min(data[:, 1])
    # print(dx, dy, end = ' ')
    # print(pca.explained_variance_ratio_[0])
    
    # limit: Minimun value of first principal compnent, which is in the 
    # range of [0.9 - 0.99]
    # The shorter the star is, the smaller the requirement for 
    # the size of the first principal component is.
    limit = 0.8 + 0.004 * (dx + dy)

    # limit = 0.5


    if limit > 0.995:
        limit = 0.995
    # The first principal component is greater than the limit.
    # print(pca.explained_variance_ratio_[0], limit)
    if pca.explained_variance_ratio_[0] > limit: 
        # print(pca.explained_variance_ratio_[0], limit, 0.8 + 0.006 * (dx + dy))
        return k, b, pca.explained_variance_ratio_[0]
    return 0, 0, 0 # Directly identified as a fake star.

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
    th, cnt = 0, 0
    for k in range(256):
        if hist[255 - k] != 0:
            n = n - 1
            cnt += hist[255 - k]
            if n <= 0 and cnt > 20 :
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
    '''Calculate the mean value by nearest 80% values.
    
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

def isVisited(Vis, x, y):
    '''Figure out whether the current window is visited.'''
    for v in Vis:
        dis = np.abs(x - v[0]) + np.abs(y - v[1])
        if dis < 30:
            Vis.append([x, y])
            return True
    Vis.append([x, y])
    return False

def Direction_estimate(image):
    '''Estimate the direction of a single image.'''
    
    # Size of window and image.
    winsize = 100 
    rowsize, colsize = image.shape
    
    # Result.
    Theta, Intercept, Linear, Window = list([]), list([]), list([]), list([])
    show = Image.fromarray(image)
    # font = ImageFont.truetype('C:\\Windows\\Fonts\\SIMYOU.TTF', 32)
    font = ImageFont.truetype('arial.ttf', 32)
    anno = ImageDraw.Draw(show)

    Vis = []
    x, y = 0, 0
    for i in range(rowsize // winsize):
        for j in range(colsize // winsize):
            # A window of star image.
            window = image[i * winsize : (i + 1) * winsize, j * winsize : (j + 1) * winsize]
            mean = np.mean(window)

            # Skip while it's too bright.
            if mean > 180:  
                continue
            
            # Threshoulding.
            thImg = threshold(window, 5) 
            
            # Get the coordinates of positive points.
            points = np.array(np.where(thImg == 255))


            # Skip while the distribution is too scattered.
            # if points.shape[0] > 0.1 * winsize * winsize:
            #     continue

            if np.std(points[0]) + np.std(points[1]) > 30: 
                continue
            # print(i, j, np.std(points[0]) + np.std(points[1]))

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
                thImg = threshold(window, 5) 
                
                # Get the coordinates of positive points.
                points = np.array(np.where(thImg == 255))

            # image[list((points.T + [i * winsize + dx, j * winsize + dy]).T)] = 255

            # Skip visited window
            x, y = np.mean(points, 1).astype(np.int32) + [i * winsize + dx, j * winsize + dy]
            if isVisited(Vis, x, y):
                continue

            # Coordinate transformation.
            coordins = np.vstack((points[1], winsize - 1 - points[0])) 
            
            # Filter out the noise and seperate different star in the window.
            groups = Cluster(coordins) 

            for group in groups:
                theta, intercept, linear = pca(group, winsize * j + dy, image.shape[0] - winsize * (i + 1) - dx)
                # (0, 0) indicates that there are no stars in the window.
                if theta != 0:
                    Theta.append(theta)
                    Intercept.append(intercept)
                    Linear.append(linear)
                    Window.append(window)
                    # anno.line((y + 5, x + 5, j * winsize + dy + 70,  i * winsize + dx + 70), fill = 255, width = 1)
                    # anno.text((j * winsize + dy + 70, i * winsize + dx + 70), '{:.4f}'.format(linear), font = font, fill = 'white')
                    # print(points)C:\\Windows\\Fonts\\SIMYOU.TTF
                    # print(list((points.T + np.array([i * winsize + dx, j * winsize + dy])).T))
                    # image[list((points.T + [i * winsize + dx, j * winsize + dy]).T)] = 255
                    # image[list((points.T + np.array([i * winsize + dx, j * winsize + dy])).T)] = 255
            #     print(len(list(points.T)), np.arctan(theta) * 180 / np.pi, linear)
                    # plt.figure()
                    # plt.imshow(np.hstack((thImg, window)), cmap = 'gray')
                    # plt.show()
    # plt.figure()
    # plt.imshow(image, cmap = 'gray')
    # plt.show()
    if len(Theta) < 2:
        return [999999999, 999999999]
    # print(Theta)

    Theta = np.vstack((Theta, Intercept, Linear)).T

    Theta = np.array(sorted(Theta, key=lambda x: x[2]))
    # print(Theta)
    Linear = Theta[:, 2]
    Intercept = Theta[:, 1]
    Theta = Theta[:, 0]
    # print(len(Theta))
    num = len(Theta) // 1000

    Theta = Theta[num: ]
    Intercept = Intercept[num: ]
    Linear = Linear[num: ]
    Res = []
    for i in range(len(Theta)):
        for j in range(i + 1, len(Theta)):
            k1, k2 = Theta[i], Theta[j]
            b1, b2 = Intercept[i], Intercept[j]
            if k1 == k2:
                continue
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
            # print(k1, b1, k2, b2, x, y)
            # Res.append([x, y, abs(b1 - b2)])
            if np.linalg.norm([x - image.shape[0] // 2, y - image.shape[1] // 2]) > 5 * np.linalg.norm(image.shape):
                if x > 0:
                    x, y = -x, -y
            Res.append([x, y])

    # print(Res)
    # print(Res, sep='\n')
    Res = np.array(Res)
    Use = Res
    pos = Res[np.where(Res[:, 0] >  0)]
    neg = Res[np.where(Res[:, 0] <= 0)]

    # print(pos)
    # print(neg)
    if pos.shape[0] >= neg.shape[0]:
        Use = pos
    else:
        Use = neg
    # print(Use.shape)
    # print(np.mean(Use, 0))
    # print(Use)

    # print('dis: ')
    dis = np.linalg.norm(Use - np.array(image.shape) / 2, axis=1)
    # print(np.linalg.norm(Use - np.array(image.shape) / 2, axis=1))
    inf = np.sum(dis > 8192)

    if inf < Use.shape[0] // 3:
        for i in range(Use.shape[0] // 3):
            mean = np.mean(Use, 0)
            diff = np.sum((Use - mean) * (Use - mean), 1)
            var = np.sum(diff) / Use.shape[0]

            # if var > 100:
            #     break
            # print(var)

            Use = np.delete(Use, np.argmax(diff), axis=0)
    S = np.mean(Use, 0)
    '''
    A = - np.ones((len(Theta), 2))
    A[:, 0] = np.array(Theta)
    b = - np.array(Intercept)
    S = np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(b)
    '''

    # S[0], S[1] = image.shape[0] - S[1], S[0]
    S[0] = image.shape[0] - S[0]
    # print(np.arctan(S[1] / S[0]) * 180 / np.pi)
    return S























    # print(np.array(Res))
    # Res = np.array(sorted(Res, key=lambda x: x[2]))
    Res = np.array(sorted(Res, key=lambda x : abs(x[0] - image.shape[0] / 2) 
                                            + abs(x[1] - image.shape[1] / 2) )) # [: len(Res) - len(Res) // 3]
    # Use = Res[: len(Res) - len(Res) // 3]
    Use = Res
    # print(Res[:, 1] / Res[:, 0])

    # print(Res)
    pos = Use[np.where(Use[:, 0] >  0)]
    # pos = np.array(sorted(pos, key=lambda x: x[2]))[len(pos) // 3:]
    neg = Use[np.where(Use[:, 0] <= 0)]
    # neg = np.array(sorted(neg, key=lambda x: x[2]))[len(neg) // 3:]
    # print(pos)
    # print(neg)
    # print(np.mean(pos, 0))
    S = []
    if pos.shape[0] >= neg.shape[0]:
        S = np.mean(pos, 0)
    else:
        S = np.mean(neg, 0)
    # print(np.mean(pos, 0))
    # print(np.mean(neg, 0))
    print(S)

    dis = np.linalg.norm(S - np.array(image.shape))
    print('dis: ', dis)
    if dis > 3 * np.linalg.norm(np.array(image.shape) - [0, 0]):
        Use = Res[len(Res) // 3: ]
        pos = np.mean(Use[np.where(Use[:, 0] > 0)], 0)
        neg = np.mean(Use[np.where(Use[:, 0] <= 0)], 0)
        if abs(pos[0]) + abs(pos[1]) > abs(neg[0]) + abs(neg[1]):
            S = pos
        else:
            S = neg
    # print(np.arctan(S[1] / S[0]) * 180 / np.pi)
    # The least squares solution of overdetermined equations: AX = b
    # For y = kx + b => kx - y = -b
    # b = [-b, ..., -b]
    # A = - np.ones((len(Theta), 2))
    # A[:, 0] = np.array(Theta)
    # b = - np.array(Intercept)
    # S = np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(b)
    # print(S)
    # anno.ellipse((S[0] - 5, (image.shape[0] - S[1]) - 5, S[0] + 5, (image.shape[0] - S[1]) + 5), fill = 'white')
    # show.show()

    print(np.arctan(S[0]/S[1]) * 180 / np.pi)
    return S[: 2]













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
    # the star point is about -90~90, and the mean value requires special treatment.
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


def get_mse(real, predict):
    """Mean square error."""
    predict = np.array(predict)
    diff = np.linalg.norm(predict - real, axis=1)
    # print(diff * diff)
    # print(np.mean(diff * diff))
    return np.sqrt(sum(diff * diff) / (len(predict) - 1))
    # return sum([(pred - real) ** 2 for pred in predict]) / (len(predict) - 1)

def get_std(predict):
    predict = np.array(predict)
    return np.std(predict)

if __name__ == '__main__':
    file_path = r'./graph/'
    images = os.listdir(file_path)
    Center = []
    for image in images:
        if image[-3:] != 'png':
            continue
        src = cv2.imread(file_path + image, 0)
        print(file_path + image)
        print(src.shape)

        blured = cv2.blur(src, (3, 3))
        center =  Direction_estimate(blured)
        Center.append(center)
        print(center)
    # mse = get_mse([1024, 1024], Center)
    # mse = get_std(Center)
    