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
    filename = r'./graph/test_5_3.png'
    src = cv2.imread(filename, 0)
    
    # 估计旋转中心
    src = cv2.blur(src, (3, 3))
    rot_center = est.Direction_estimate(src)
    print('Rcenter:', rot_center)
    # rot_center = [999999999, 999999999]

    # 星点提取、质心定位
    retImg, centers, cnt = ext.extract(src.copy(), rot_center)
    centers = centers[:cnt]
    print('Stars:', cnt)

    # plt.figure()
    # plt.imshow(retImg, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # 星图识别、姿态解算
    att, starId = iden.identify(centers)
    print('Attitude:', att)
    
    starId = starId[np.where(starId >= 0)]
    # print(starId[np.where(starId >= 0)])

    # 重投影
    retImg, iden = iden.reProjection(retImg, att, centers)
    print('Identified:', iden)

    # 显示
    plt.figure()
    plt.imshow(retImg)
    # if rot_center[0] != 999999999:
    #     plt.scatter([rot_center[0]], [rot_center[1]], color='red')
    #     plt.plot([rot_center[0], rot_center[0]], [rot_center[1] - 200, rot_center[1] + 200], color='red')
    #     plt.plot([rot_center[0] - 200, rot_center[0] + 200], [rot_center[1], rot_center[1]], color='red')
    plt.show()