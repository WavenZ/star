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
    '''通过主成分分析计算星点的运动角度
        
    参数：
        points: 星点坐标，shape = (2, n)
        xref: 窗口左上角的 x 坐标
        yref: 窗口右上角的 y 坐标
    
    返回值:
        返回星点所在位置直线的参数 (k, b) 或者 (0, 0)
    '''
    # 转置一下
    data = points.T
    
    # 主成分分析
    pca = PCA()
    pca.fit(data)
    if pca.components_[0, 0]: # 防止分母为0 
        k = pca.components_[0, 1] / pca.components_[0, 0]
    else:
        k = 999999
    if k * (xref + np.mean(points[0])): # 防止分母为0
        b = np.mean(yref + points[1]) + 1 / k * (xref + np.mean(points[0]))
    else:
        b = 999999
    
    # dx, dy 为所有点的 x 和 y 坐标范围，用以辅助设置第一主成分的限制（下限）
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
        return k, b
    return 0, 0 # 直接判定为非星点

def threshold(img, n):
    """阈值化

    根据灰度分布直方图，找到窗口中前 n 个灰阶的点
    
    参数: 
        img: 图像
        n: 阈值化之后为 255 的灰阶数

    返回值:
        阈值化之后的图像
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
    '''聚类，用于去除远离星点噪声

    参数:
        coordins: 坐标，shape = (2, n)
    
    返回值: 
        聚类的结果（若干个组）
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
    '''求均值，只取靠得最近的前 80% 的值用来求均值
    
    参数: 
        values: 要求的均值的集合
    
    返回值:
        均值
    '''
    if len(values) < 2:
        return np.mean(values)
    # 去均值化
    diff = np.abs(values - np.mean(values))
    # 根据误差排序
    data = np.vstack((diff, values)).T
    data = data[np.argsort(data[:,0])]
    return np.mean(data[:round(len(data) * 0.8), 1])

def mean_value2(values, limit):
    '''求均值

    通过限定方差的大小来滤除一些无效数据

    参数:
        values: 要计算均值的数据
        limit: 限定的方差大小
    
    返回值:
        经过方差滤除之后的数据的均值
    '''

    # 数据量小于等于 2, 直接返回均值
    n = len(values)
    if n <= 2:
        return np.mean(values)
    
    # 先进行排序
    values = np.sort(values)
    std = np.std(values)
    # 迭代计算
    while len(values) >= max(2, 0.7 * n) or std > limit:
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
    '''对单张星图进行星点运动的方向估计'''
    
    # 窗口大小
    winsize = 100 
    rowsize, colsize = image.shape
    
    # 结果向量
    Theta = list([])

    for i in range(rowsize // winsize):
        for j in range(colsize // winsize):
            # 星图中的一个窗口
            window = image[i * winsize : (i + 1) * winsize, j * winsize : (j + 1) * winsize] 
            mean = np.mean(window)
            # 均值过大（太亮），跳过
            if mean > 180: 
                continue
            
            # 阈值化图像
            thImg = threshold(window, 6) 
            
            # 取阈值化图像中亮点的坐标
            points = np.array(np.where(thImg == 255)) 
            # 亮点分布太分散，跳过
            if np.std(points[0]) + np.std(points[1]) > 50: 
                continue

            # 坐标转化
            coordins = np.vstack((points[1], winsize - 1 - points[0])) 
            
            # 去除噪声，同时分离窗口中的多个星点
            groups = Cluster(coordins) 

            # plt.figure()
            # plt.imshow(thImg, cmap = 'gray')
            # plt.show()
            for group in groups:
                # 主成分分析计算角度
                theta = pca(group, winsize * j, image.shape[0] - winsize * (i + 1))
                # (0, 0) 表示图中无星点
                if theta != (0, 0):
                    Theta.append(theta[0])

    # print(np.arctan(np.array(Theta)) * 180 / np.pi)
    
    # 去除非法值
    for i in range(len(Theta)):
        if Theta[i] == float('inf'):
            Theta[i] = -99999

    # 如果一个星点都没估计出，直接返回
    if len(Theta) == 0: 
        return 0

    # 如果大于10，则表示星点运动方向在 ±90 左右，求均值需要特殊处理
    if np.mean(np.abs(Theta)) > 10: 
        res = np.arctan(Theta) * 180 / np.pi
        res = list(map(lambda x : x if x > 0 else 180 + x, res))
        res = mean_value2(res, 5)
        if res > 90:
            res = res - 180
    else:
        res = np.arctan(mean_value2(np.array(Theta), 2)) * 180 / np.pi
    # 方向的估计值
    return res 

def imshow(*images):
    '''显示星图'''

    for image in images:
        plt.figure()
        plt.imshow(image, cmap='gray')
    plt.show()

def Kernel(size, width, theta):
    """构造卷积核
        
    参数: size：卷积核尺寸
          width：正区域宽度
          theta：角度
    注：(size - width)最好是偶数，以使得卷积核对称
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