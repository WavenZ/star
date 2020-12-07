# -*- encoding: utf-8 -*-
'''
    @File    :   main.py
    @Time    :   2020/12/07 06:55:46
    @Author  :   wavenz 
    @Version :   1.0
    @Contact :   1435595081@qq.com
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

import estimate.est as est
import extract.extract as ext
import identifier.identify as iden

from cv2 import cv2

import os
import time

if __name__ == "__main__":

    # 读文件
    # filename = r'./zl/9-2.png'
    filename = r'./graph/test_5_1.png'
    src = cv2.imread(filename, 0)[:2048, :2048]

    # 估计旋转中心
    src = cv2.blur(src, (3, 3))
    rot_center = est.Direction_estimate(src)
    print('Rcenter:', rot_center)
    # rot_center = [999999999, 999999999]

    # 星点提取、质心定位
    retImg, centers, cnt = ext.extract(src.copy(), rot_center)
    centers = centers[:cnt]
    print('Stars:', cnt)

    # 星图识别、姿态解算
    att, starId = iden.identify(centers)
    print('Attitude:', att)
    
    starId = starId[np.where(starId >= 0)]

    # 重投影
    retImg, iden = iden.reProjection(retImg, att, centers)
    print('Identified:', iden)

    # 显示
    plt.figure()
    plt.imshow(retImg)
    plt.show()