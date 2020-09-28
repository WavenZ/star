import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import os

def pca(points, xref, yref):
    '''
    通过主成分分析计算星点的运动角度
        points: 星点坐标，shape = (2, n)
        xref: 窗口左上角的x坐标
        yref: 窗口右上角的y坐标
    '''
    data = points.T
    pca = PCA()   # 保留所有成分
    pca.fit(data)
    if pca.components_[0, 0]: # 防止分母为0 
        k = pca.components_[0, 1] / pca.components_[0, 0]
    else:
        k = 999999
    if k * (xref + np.mean(points[0])): # 防止分母为0
        b = np.mean(yref + points[1]) + 1 / k * (xref + np.mean(points[0]))
    else:
        b = 999999
    # print(data)
    # dx, dy 为所有点的x和y坐标范围，用以辅助设置第一主成分的限制（下限）
    dx = np.max(data[:, 0]) - np.min(data[:, 0])
    dy = np.max(data[:, 1]) - np.min(data[:, 1])
    # print(dx, dy, end = ' ')
    # print(pca.explained_variance_ratio_[0])
    # limit 第一主成分的最小值，范围为[0.9 - 0.99]
    # 星点越短，则对第一主成分大小的要求减小
    limit = 0.9 + 0.003 * (dx + dy)
    if limit > 0.99:
        limit = 0.99
    if pca.explained_variance_ratio_[0] > limit: # 第一主成分大于limit
        # print(pca.explained_variance_ratio_[0])
        return np.array([k, b, pca.explained_variance_ratio_[0]])
    else:
        return np.array([0, 0, 0]) # 直接判定为非星点
def threshold(img, n):
    """
        阈值化
        方法：根据灰度分布直方图，找到窗口中前n个灰阶的点
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
    ret, thImg = cv2.threshold(img,th,255,cv2.THRESH_BINARY)
    return thImg
def Cluster(coordins):
    '''
    聚类，用于去除远离星点噪声
        coordins: 坐标，shape = (2, n)
    '''
    data = coordins.T
    # eps:可以看做类的最大间距
    # min_samples:类的最小点数
    pred = DBSCAN(eps = 20, min_samples = 5).fit_predict(data)
    groups = []
    for i in np.unique(pred):
        cnt = list(pred).count(i)
        if cnt > 0.9 * pred.size: # 单个类的点数大于总点数的90%
            groups.append(coordins[:, np.where(pred == i)].squeeze(1))
    return groups
def mean_value(values):
    '''
    求均值，只取靠得最近的前80%的值用来求均值
        values: 值
    '''
    if len(values) < 2:
        return np.mean(values)
    diff = np.abs(values - np.mean(values))
    data = np.vstack((diff, values)).T
    data = data[np.argsort(data[:,0])]
    # print(np.arctan(data) * 180 / np.pi)
    # print(np.arctan(data[:round(len(data) * 0.8)]) * 180 / np.pi, end = ' ')
    return np.mean(data[:round(len(data) * 0.8), 1])
def mean_value2(directions, limit):
    if len(directions) <= 2:
        return np.mean(directions)
    directions = np.sort(directions)
    std = np.std(directions)
    for i in range(len(directions)):
        # print(directions)
        if std > limit:
            stda = np.std(directions[1:])
            stdb = np.std(directions[:-1])
            if stda < stdb:
                std = stda
                directions = directions[1:]
            else:
                std = stdb
                directions = directions[:-1]
        else:
            break
    return np.mean(directions)
def Direction_estimate(image):
    '''
    对单张星图进行星点运动的方向估计
    '''
    row, col = image.shape
    win = 100 # 窗口大小
    Theta = list([])
    for i in range(row // win):
        for j in range(col // win):
            window = image[i * win : (i + 1) * win, j * win : (j + 1) * win] # 星图中的一个窗口
            mean = np.mean(window)
            if mean > 180: # 太亮，跳过
                continue
            thImg = threshold(window, 8) # 阈值化图像
            points = np.array(np.where(thImg == 255)) # 取阈值化图像中亮点的坐标
            if np.std(points[0]) + np.std(points[1]) > 50: # 亮点分布太分散，跳过
                continue
            coordins = np.vstack((points[1], win - 1 - points[0])) # 转化一下
            groups = Cluster(coordins) # 去除噪声，同时单个窗口可能存在多个星点
            # plt.figure()
            # plt.imshow(thImg, cmap = 'gray')
            # plt.show()
            for group in groups:
                # print(group)
                theta = pca(group, win * j, image.shape[0] - win * (i + 1))
                if theta[0] != 0:
                    Theta.append(theta[[0, 2]])
                    # plt.figure()
                    # plt.imshow(thImg, cmap = 'gray')
    print(Theta, sep='\n')
    return 0
    for i in range(len(Theta)):
        if Theta[i][0] == float('inf'):
            Theta[i][0] = -99999
    if len(Theta) == 0: # 如果一个星点都没估计出，直接返回
        return 0
    if np.mean(np.abs(Theta[:, 0])) > 10: # 如果大于10，则表示星点运动方向在±90左右，求均值需要特殊处理
        res = np.arctan(Theta[:, 0]) * 180 / np.pi
        res = list(map(lambda x : x if x > 0 else 180 + x, res))
        res = mean_value2(res, 5)
        if res > 90:
            res = res - 180
    else:
        res = np.arctan(mean_value2(np.array(Theta), 2)) * 180 / np.pi
   
    return res # 方向的估计值

if __name__ == '__main__':
    file_path = r'C:\Users\14355\star\image\b\5_2_72.png'
    img = cv2.imread(file_path, 0)
    blured = cv2.blur(img, (3, 3))
    direc = Direction_estimate(blured)
    print(direc)
    