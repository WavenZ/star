# -*- encoding: utf-8 -*-
'''
@File    :   useless.py
@Time    :   2020/12/07 06:57:48
@Author  :   wavenz 
@Version :   1.0
@Contact :   1435595081@qq.com
'''

import matplotlib.pyplot as plt
import generator.generate_starImg as ggs

from PIL import Image

import cv2
from cv2 import cv2

import numpy as np
if __name__ == "__main__":
    
    # # Attitude: (yaw, pitch, roll)
    # att = [20, 40, 60]

    # # Angle Velocity: (yaw, pitch, roll)
    # dps = [5, 5, 5]

    # # Generate
    # retImg = ggs.genDynamic(att, dps, 100)

    # # Show
    # plt.figure()
    # plt.imshow(retImg, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    res = cv2.imread('./graph/r10_5.png', 0).astype(np.int32)

    # res += (np.random.randn(2048, 2048) * 3).astype(np.int32)

    res[np.where(res < 0)] = 0
    res[np.where(res > 255)] = 255

    plt.figure()
    plt.imshow(res, cmap='gray', vmin=0, vmax=255)
    plt.show()

    res = Image.fromarray(res)
    res = res.convert('L')
    res.save('./graph/rr10_5.png')