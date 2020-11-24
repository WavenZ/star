import os
import cv2
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import numpy.ctypeslib as npct

from ctypes import *
from cv2 import cv2
from PIL import Image


# def conv(src, center):
#     '''Convolution for rotation mode.

#     '''
#     # print(center[0], center[1])
#     # center[1] = src.shape[0] - center[1]
#     conv_init()
#     ret = np.zeros_like(src)
#     start = time.time()
#     for row in range(6 , (src.shape[0] - 5)):
#         for col in range(6 , (src.shape[1] - 5)):
#             row = src.shape[0] - row - 1
#             print((np.arctan((col - center[0]) / (row - center[1])) * 180 / np.pi).astype(np.int).item() + 90, end=' ')
#             kernel = kernels[(np.arctan((col - center[0]) / (row - center[1])) * 180 / np.pi).astype(np.int).item() + 90]
#             conv = np.sum(kernel * src[row - 5:row + 6, col - 5:col + 6])
#             ret[row, col] = conv if conv >= 0 else 0
#         break
#     end = time.time()
#     print(end - start)
#     return ret

def conv(src, center):
    '''Convolution for rotation mode.

    Call conv.exe, implement by cpp.
    '''
    print(src.dtype)
    img2txt(src)
    os.system('conv {} {} {} {}'.format(src.shape[0], src.shape[1], center[0], center[1]))
    ret = txt2img('img1.txt', src.shape[0], src.shape[1])
    
    return ret

def get_kernel(size, width, theta):
    """Construct the convolution kernel.

    Args: size：size of kernel, (Height, Width)
          width：Width of the positive region.
          theta：Rotation angle.

    Notes：
        (size - width) shoule be even, so that the converlution kernel is symmetric.
    """

    temp = np.zeros((size + 20, size + 20))
    temp [(size - width) // 2 + 10: (size - width) // 2 + 10 + width,:] = 1
    temp = Image.fromarray(temp)
    temp = temp.rotate(theta)
    temp = np.array(temp)
    kernel = temp[10:-10, 10:-10]
    cnt = np.sum(kernel)
    for i in range(size):
        for j in range(size):
            kernel[i, j] = cnt / (cnt - size * size) if kernel[i, j] == 0 else kernel[i, j]
    kernel = kernel / 4
    return kernel

def threshold(img, percentage = None, num = None):
    """
        阈值化
        方法：根据灰度分布直方图，找到窗口中最亮的若干点
    """
    if num is None and percentage is not None:
        num = img.size * percentage
    if num is None and percentage is None:
        num = img.size * 0.1
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    cnt = 0
    th = 0
    for k in range(256):
        cnt = cnt + hist[255 - k]
        if cnt > num:
            th = 255 - k - 1
            break
    _, thImg = cv2.threshold(img,th,255,cv2.THRESH_BINARY)
    return thImg

def enhance(image, center):
    '''The image is enhanced by convolution.

    Args:
        image: Image to be enhanced.
        center: Center of rotation.

    Returns:
        Enhanced image
    '''
    # mode: 0 parallel mode, 1 ratation mode
    mode = 0

    # Select mode.
    dis = np.linalg.norm(center - np.array(image.shape))
    if dis < 5 * np.linalg.norm(np.array(image.shape) - [0, 0]):
        mode = 1
    if mode == 0:
        theta = np.arctan(- center[0] / center[1]) * 180 /  np.pi
        kernel = get_kernel(17, 3, theta)
        ret = cv2.filter2D(image, -1, kernel)
    if mode == 1:
        ret = conv(image, center)

    return ret

def img2txt(image):
    with open('img.txt', 'w', newline='') as f:
        s = ''
        for row in image:
            s += (''.join(chr(pixel) for pixel in row))
        f.write(s)  

def txt2img(filename, h, w):
    buf = []
    with open(filename, 'rb') as f:
        data = f.read()
        for i in range(h):
            temp = []
            temp = [(data[i * w + j]) for j in range(w)]
            buf.append(temp)

    return np.array(buf).astype('uint8')


def connectedComponents(image):
    start = time.time()
    # ans = list([])
    # src = image.copy()
    # Q = collections.deque()
    # for i in range(src.shape[0]):
    #     for j in range(src.shape[1]):
    #         if src[i][j] == 255:
    #             src[i][j] = 0
    #             Q.append([i, j])
    #             temp = list([])
    #             while Q:
    #                 for k in range(len(Q)):
    #                     x, y = Q.popleft()
    #                     temp.append([x, y])
    #                     for dx in [-1, 0, 1]:
    #                         for dy in [-1, 0, 1]:
    #                             if dx == 0 and dy == 0:
    #                                 continue
    #                             if x + dx < 0 or x + dx >= src.shape[0]:
    #                                 continue
    #                             if y + dy < 0 or y + dy >= src.shape[1]:
    #                                 continue
    #                             if src[x + dx][y + dy] == 255:
    #                                 src[x + dx][y + dy] = 0
    #                                 Q.append([x + dx, y + dy])
    #             ans.append(temp)

    ans, temp = list([]), list([])
    src = image.copy()
    def dfs(x, y):
        # print(x, y)
        nonlocal temp, src
        temp.append([x, y])
        src[x][y] = 0
        if x > 0 and src[x - 1][y] == 255:
            dfs(x - 1, y)
        if x < src.shape[0] - 1 and src[x + 1][y] == 255:
            dfs(x + 1, y)
        if y > 0 and src[x][y - 1] == 255:
            dfs(x, y - 1)
        if y < src.shape[1] - 1 and src[x][y + 1] == 255:
            dfs(x, y + 1)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i][j] == 255:
                temp = []
                dfs(i, j)
                ans.append(temp)
    end = time.time()
    print(end - start)
    return ans


def extract(src, theta):
    '''Extract stars from src(image).

    Call function conv_and_bin() in extract.dll

    Args:
        src: Source image.
        center: Rotation center.

    Returns:
        Binarized image.
    '''

    kernels = []
    for alpha in range(-90, 91):
        kernels.append(get_kernel(13, 3, alpha))

    kernels = np.array(kernels).astype(np.float)

    # x0, y0 = c_double(theta[0]), c_double(theta[1])
    x0, y0 = theta[0], theta[1]
    cnt = c_int(int(0))

    res = np.zeros_like(src)
    centers = np.zeros((1000, 3)).astype(np.float)

    # lib = npct.load_library("test",".")
    lib = cdll.LoadLibrary('./extract/conv.so')
    lib.conv_and_bin.argtypes= [npct.ndpointer(dtype=np.uint8, ndim=src.ndim, shape=src.shape, flags="C_CONTIGUOUS"),
        c_int, c_int, c_double, c_double, npct.ndpointer(dtype=np.uint8, ndim=res.ndim, shape=res.shape, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype=np.float, ndim=kernels.ndim, shape=kernels.shape, flags="C_CONTIGUOUS"),
        npct.ndpointer(dtype=np.float, ndim=centers.ndim, shape=centers.shape, flags="C_CONTIGUOUS"),
        POINTER(c_int)]

    lib.conv_and_bin(src, c_int(src.shape[0]), c_int(src.shape[1]), c_double(x0), c_double(y0), res, kernels, centers, pointer(cnt))
    return res, centers.copy(), cnt.value

if __name__ == "__main__":
    kernels = []
    for alpha in range(-90, 91):
        kernels.append(get_kernel(13, 3, alpha))
    plt.figure()
    plt.imshow(kernels[90], cmap='gray')
    plt.show()