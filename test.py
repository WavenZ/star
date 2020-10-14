import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2
from PIL import Image

kernels = []

def get_kernel(size, width, theta):
    """Construct the convolution kernel.
        
    Args: size：size of kernel, (Height, Width)
          width：Width of the positive region.
          theta：Rotation angle.
    
    Notes：
        (size - width) shoule be even, so that the converlution kernel is symmetric.
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


def conv_init():
    for theta in range(-90, 91):
        kernels.append(get_kernel(11, 3, theta))

def kernels2txt():
    with open('kernel.txt', 'w') as f:
        for kernel in kernels:
            for row in kernel:
                for elem in row:
                    f.write(str(elem) + ' ')
                f.write('\n')

def img2txt(image):
    with open('img.txt', 'w', newline='') as f:
        # f.write((str(image.shape[0]) + '\n'))
        # f.write((str(image.shape[1]) + '\n'))
        s = ''
        for row in image:
            s += (''.join(chr(pixel) for pixel in row))
            # for pixel in row:
                # s += chr(pixel)
        print(f.write(s))

def txt2img(filename, h, w):
    buf = []
    with open(filename, 'rb') as f:
        # h = int(f.readline())
        # w = int(f.readline())
        # print(h, w)
        # print(f.read())
        data = f.read()
        print(type(data))
        print(len(data))
        for i in range(h):
            temp = []
            temp = [(data[i * w + j]) for j in range(w)]
            buf.append(temp)
    return np.array(buf)

def conv(src, center):
    '''Convolution for rotation mode.

    '''
    img2txt(src)
    os.system('conv {} {} {} {}'.format(src.shape[0], src.shape[1], center[0], center[1]))
    ret = txt2img('img1.txt', src.shape[0], src.shape[1])
    return ret

if __name__ == "__main__":
    # conv_init()
    # kernels2txt()

    img = cv2.imread('./graph/5.png', 0)
    img = conv(img, [1024, 1024])

    # img = txt2img('img.txt')
    # print(img[0])
    plt.figure()
    print(img.shape, img.shape)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()