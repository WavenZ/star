import cv2
from ctypes import *
import time
from PIL import Image
import numpy as np
import numpy.ctypeslib as npct
import matplotlib.pyplot as plt

kernels = []

def conv_init():
    for theta in range(-90, 91):
        kernels.append(get_kernel(11, 3, theta))

def get_kernel(size, width, theta):

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
    conv_init()
    src = cv2.imread('./graph/1.png', 0).astype(np.uint8)
    res = np.zeros_like(src)

    kernels = np.array(kernels).astype(np.float)

    x0, y0 = 1024, 1024
    # lib = npct.load_library("test",".")
    lib = cdll.LoadLibrary('./conv.so')
    # lib.onedemiarr.argtypes = [npct.ndpointer(dtype = np.int, ndim = 1, flags="C_CONTIGUOUS"), c_int]
    lib.conv_and_bin.argtypes = [npct.ndpointer(dtype=np.uint8, ndim=src.ndim, shape=src.shape, flags="C_CONTIGUOUS"),
        c_int, c_int, c_int, c_int, npct.ndpointer(dtype=np.uint8, ndim=res.ndim, shape=res.shape, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype=np.float, ndim=kernels.ndim, shape=kernels.shape, flags="C_CONTIGUOUS")]
    
    start = time.time()
    lib.conv_and_bin(src, c_int(src.shape[0]), c_int(src.shape[1]), c_int(x0), c_int(y0), res, kernels)
    end = time.time()
    print('Time cost:', end - start)

    plt.figure()
    plt.imshow(res, cmap='gray', vmin=0, vmax=255)
    plt.show()    