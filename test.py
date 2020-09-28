import matplotlib.pyplot as plt
import numpy as np
import angest.est as ae
import cv2
import os

from cv2 import cv2

def get_mse(real, predict):
    """
    均方误差 估计值与真值 偏差
    """
    return sum([(pred - real) ** 2 for pred in predict]) / len(predict)


if __name__ == "__main__":

    file_path = r'./graph/dynamic/5dps/100ms/30/'
    images = os.listdir(file_path)
    directions = []
    for image in images:
        src = cv2.imread(file_path + '\\' + image, 0)
        # src = cv2.blur(src, (3, 3))
        direction = ae.Direction_estimate(src)
        print(direction)
        directions.append(direction)
    # print('Res: ', np.array(directions))
    print()
    print('[Min] ', np.min(directions))
    print('[Max] ', np.max(directions))
    print('[Avg] ', np.mean(directions))  
    print('[Mse] ', get_mse(60, directions))

    x = np.linspace(0, len(images), len(images), endpoint=False)
    plt.figure()
    plt.ylim((-1, 1))
    plt.scatter(x, np.array(directions) - 60)
    plt.show()