import matplotlib.pyplot as plt
import numpy as np
import estimate.est as ae
import extract.extract as ex
import generator.gen_dynamic as gd
import identifier.pyramid as ip
import cv2
import os
import time

from cv2 import cv2

def get_mse(real, predict):
    """Mean square error."""

    return sum([(pred - real) ** 2 for pred in predict]) / len(predict)

if __name__ == "__main__":

    # 读文件
    filename = r'./graph/test_10_5.png'
    src = cv2.imread(filename, 0)
    
    # 估计旋转中心
    src = cv2.blur(src, (3, 3))
    rot_center = ae.Direction_estimate(src)
    print('Rcenter:', rot_center)
    # rot_center = [999999999, 999999999]
    # 星点提取、质心定位
    retImg, centers, cnt = ex.extract(src.copy(), rot_center)
    centers = centers[:cnt]
    print('Stars:', cnt)

    # 星图识别、姿态解算
    att = ip.identify(centers)
    print('Attitude:', att)
    
    # 重投影
    retImg, iden = ip.reProjection(retImg, att, centers)
    print('Identified:', iden)
    plt.figure()
    plt.imshow(retImg)
    plt.show()