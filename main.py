import matplotlib.pyplot as plt
import numpy as np
import estimate.est as ae
import generator.gen_dynamic as gd
import cv2
import os

from cv2 import cv2

def get_mse(real, predict):
    """Mean square error."""

    return sum([(pred - real) ** 2 for pred in predict]) / len(predict)


if __name__ == "__main__":

    # file_path = r'./graph/dynamic/5dps/100ms/30/'
    # file_path = r'../graph/dynamic/variable_dps/6dps'
    # file_path = r'../graph/dynamic/fix0'
    file_path = r'./graph/'
    images = os.listdir(file_path)
    directions = []
    for image in images:
        src = cv2.imread(file_path + '\\' + image, 0)
        src = cv2.blur(src, (3, 3))
        direction = ae.Direction_estimate(src)
        print(image, direction)
        directions.append(direction)
    # print('Res: ', np.array(directions))
    print()
    print('[Min] ', np.min(directions))
    print('[Max] ', np.max(directions))
    print('[Avg] ', np.mean(directions))
    print('[Mse] ', get_mse(60, directions))

    plot = False
    if plot:
        x = np.linspace(0, len(images), len(images), endpoint=False)
        plt.figure()
        plt.ylim((-1, 1))
        plt.scatter(x, np.array(directions) - 60)
        plt.show()

    # G = gd.StarGenerator('sao60')
    # pitch, yaw, roll = 10, 30, 60
    # pitchspd, yawspd, rollspd = 0, 5, 0
    # img, starnum = G.generate(pitch, yaw, roll, pitchspd, yawspd, rollspd, winvisible = False)
    # print('[starnum]:', starnum)
    # plt.figure()
    # plt.imshow(img, cmap='gray', )
    # plt.show()