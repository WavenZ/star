import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


import generator.gen_static_new as gsn

from cv2 import cv2

if __name__ == "__main__":
    src = cv2.imread('./graph/10_5.png', 0)
    mean = np.mean(src)
    var = np.std(src)
    print(mean, var)
    up = mean * np.ones((1024, 2048)) + np.random.randn(1024, 2048) * var
    src = np.vstack((up, src))
    # plt.figure()
    # plt.imshow(src, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    img = Image.fromarray(src)
    img = img.convert('L')
    img.save('./graph/test_10_5.png')