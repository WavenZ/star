import numpy as np
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

if __name__ == "__main__":
    conv_init()
    kernels2txt()
