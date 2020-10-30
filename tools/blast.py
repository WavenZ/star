import cv2
import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2
def blast(filename, brightness, outname):
    '''Increase the brightness of the image.

    '''
    img = cv2.imread(filename, 0)
    img += brightness
    plt.imsave(outname, img, cmap='gray', vmin=0, vmax=255)

if __name__ == "__main__":
    blast('./graph/a.png', 10, './out1.png')