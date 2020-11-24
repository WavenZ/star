import numpy as np
import matplotlib.pyplot as plt

import generator.generate_starImg as g

if __name__ == "__main__":
    
    # Attitude
    att = [30, 60, 90]

    # Generate
    retImg = g.genStatic(att)

    # Show
    plt.figure()
    plt.imshow(retImg, cmap='gray', vmin=0, vmax=255)
    plt.show()