import numpy as np
import est
import cv2
import os

if __name__ == "__main__":
    src =  r'C:\Users\14355\star\image\b\5_1_55.png'
    image = cv2.imread(src, 0)
    image = cv2.blured(image, (3, 3))
    dir = est.Direction_estimate(image)
    kernel = est.Kernel(9, 3, dir)
    ret = cv2.filter2D(image, -1, kernel)
    plt.figure()
    plt.imshow(ret, cmap = 'gray')
    plt.show()
