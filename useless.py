import numpy as np
import matplotlib.pyplot as plt

import generator.gen_static_new as gsn

if __name__ == "__main__":
    ret = gsn.genStatic([20, 30, 40])
    plt.figure()
    plt.imshow(ret, cmap='gray', vmin=0, vmax=255)
    plt.show()