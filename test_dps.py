import matplotlib.pyplot as plt
import numpy as np
import estimate.est as ae
import extract.extract as ex
import generator.gen_dynamic as gd
import cv2
import os
import time

from cv2 import cv2

def get_mse(real, predict):
    """Mean square error."""

    return sum([(pred - real) ** 2 for pred in predict]) / len(predict)

if __name__ == "__main__":

    # file_path = r'./graph/dynamic/5dps/100ms/30/'
    # file_path = r'../graph/dynamic/variable_dps/6dps'
    # file_path = r'../graph/dynamic/fix60'

    log = open('./dps_record.txt', 'a')

    curr, num = 0, 1000
    start = time.time()
    for dps in range(2, 13):
        curr = 0
        file_path = r'/media/wavenz/新加卷/graph/dps{}/'.format(dps)
        log.write('\nData: {}\n'.format(file_path))
        log.write(time.strftime('Time: %Y-%m-%d %H:%M:%S\n\n',time.localtime(time.time())))

        # file_path = r'./graph/'
        images = os.listdir(file_path)
        # print(images)
        sum_valid, sum_total = 0, 0
        for image in images:
            if image[-4:] != '.png':
                continue
            src = cv2.imread(file_path + '/' + image, 0)
            # print(src.shape)
            src = cv2.blur(src, (3, 3))
            # theta = ae.Direction_estimate(src)
            theta = [100000000, 173000000]
            # theta = [99999, 99999]
            # print(theta)
            retImg, centers, cnt = ex.extract(src.copy(), theta)
            
            # plt.imsave('./{}_extract1.png'.format(image),retImg, cmap = 'gray', vmin = 0, vmax = 255)
            reals = np.load(file_path + '/' + '{}.npy'.format(image[:-4]))
            # print('{}.npy'.format(image[:-4]))
            centers = centers[:cnt]

            valid, total = 0, reals.shape[0]
            for center in centers:
                for real in reals:
                    if np.linalg.norm((center - real)) < 5:
                        # print(center, real, np.linalg.norm((center - real)))
                        valid+=1
                        break
            sum_valid += valid
            sum_total += total
            # print(centers)
            # print(reals)
            curr+=1
            log.write('[Param = {} {:3}/{} \'{}\'   case: {:2}/{:2} = {:.3f}   total: {}/{} = {:.3f}]\n'.format(dps, curr, num, image, valid, total, valid / total, sum_valid, sum_total, sum_valid / sum_total))
            # print('[Param = {} {:3}/{} \'{}\'   case: {:2}/{:2} = {:.3f}   total: {}/{} = {:.2f}]'.format(dps, curr, num, image, valid, total, valid / total, sum_valid, sum_total, sum_valid / sum_total))
            # plt.figure()
            # plt.imshow(retImg, cmap='gray', vmin=0, vmax=255)
            # plt.show()
        print('[Param = {}  total: {}/{} = {:.3f}]'.format(dps, sum_valid, sum_total, sum_valid / sum_total))

            # plt.imsave('./graph/{}_extract.png'.format(image), np.hstack((src, retImg)), cmap = 'gray', vmin = 0, vmax = 255)
            # # print(image, direction)
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
    # plt.show()e
    log.close()