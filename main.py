import matplotlib.pyplot as plt
import numpy as np
import estimate.est as ae
import extract.extract as ex
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
    # file_path = r'../graph/dynamic/fix60'

    file_path = r'./graph/'
    images = os.listdir(file_path)
    # print(images)
    directions = []
    for image in images:
        print(image)
        src = cv2.imread('./graph/1.png', 0)
        # print(src.shape)
        src = cv2.blur(src, (3, 3))
        theta = ae.Direction_estimate(src)
        print(theta)
        ret, retImg = ex.extract(src, theta)
        plt.figure()
        plt.imshow(retImg, cmap='gray', vmin=0, vmax=255)
        plt.show()
        plt.imsave('./graph/{}_extract.png'.format(image), np.hstack((src, retImg)), cmap = 'gray', vmin = 0, vmax = 255)
        # print(image, direction)
        # directions.append(direction)
    # print('Res: ', np.array(directions))
    # print()
    # print('[Min] ', np.min(directions))
    # print('[Max] ', np.max(directions))
    # print('[Avg] ', np.mean(directions))
    # print('[Mse] ', get_mse(30, directions))

    # plot = False
    # if plot:
    #     x = np.linspace(0, len(images), len(images), endpoint=False)
    #     plt.figure()
    #     plt.ylim((-1, 1))
    #     plt.scatter(x, np.array(directions) - 30)
    #     plt.show()

    # G = gd.StarGenerator('sao60')
    # pitch, yaw, roll = 10, 30, 60
    # pitchspd, yawspd, rollspd = 0, 5, 0
    # img, starnum = G.generate(pitch, yaw, roll, pitchspd, yawspd, rollspd, winvisible = False)
    # print('[starnum]:', starnum)
    # plt.figure()
    # plt.imshow(img, cmap='gray', )
    # plt.show()