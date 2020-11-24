# -*- coding:UTF-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

from cv2 import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def put_stars(img, x0, y0, E, delta = 1.3, winvisible = False, winradius = 50):
        '''Place stars to the specified image.

        Args:
            img: Image.
            x0, y0: Coordinates of the star to be placed.
            E: Energy indensity.
            winvisible: Star highlight window.
            winradius: Radius of highlight window.
        '''
        up = int(x0) - winradius if int(x0) - winradius >= 0 else 0
        down = int(x0) + winradius + 1 if int(x0) + winradius + 1 <= img.shape[0] else img.shape[0]
        left = int(y0) - winradius if int(y0) - winradius >= 0 else 0
        right = int(y0) + winradius + 1 if int(y0) + winradius + 1 <= img.shape[1] else img.shape[1]
        x = np.linspace(up, down - 1, down - up)
        y = np.linspace(left, right - 1, right - left)
        X, Y = np.meshgrid(x, y)
        X, Y = X.T, Y.T
        if winvisible is True:
            img[up : down, left : right] += 50
        img[up : down, left : right] += E / (2 * np.pi * delta ** 2) * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * delta ** 2))

def genStatic(attitude):

    # 读星库
    catalog = np.loadtxt('./params/sao60.txt', dtype = float)

    # 各项参数
    h, w = 2048, 2048
    cx, cy, dx, dy, f = [h / 2, w / 2, 0.0055, 0.0055, 25.0]
    
    # 计算fov
    fov = np.arctan((cx * dx) / f) * 180 / np.pi * 2


    # 角度转换为弧度制
    ra, dec, rol = np.array(attitude) * np.pi / 180

    # 姿态转换矩阵：天球坐标系 -> 星敏感器坐标系
    r11 = - np.cos(rol) * np.sin(ra) - np.sin(rol) * np.sin(dec) * np.cos(ra)
    r12 = np.cos(rol) * np.cos(ra) - np.sin(rol) * np.sin(dec) * np.sin(ra)
    r13 = np.sin(rol) * np.cos(dec)
    r21 = np.sin(rol) * np.sin(ra) - np.cos(rol) * np.sin(dec) * np.cos(ra)
    r22 = - np.sin(rol) * np.cos(ra) - np.cos(rol) * np.sin(dec) * np.sin(ra)
    r23 = np.cos(rol) * np.cos(dec)
    r31 = np.cos(dec) * np.cos(ra)
    r32 = np.cos(dec) * np.sin(ra)
    r33 = np.sin(dec)

    Rbc = np.array([[r11, r12, r13], 
                    [r21, r22, r23],
                    [r31, r32, r33]])
    
    # 姿态转换矩阵：星敏感器坐标系 -> 天球坐标系
    Rcb = Rbc.T

    # 视轴指向
    S = Rcb.dot(np.array([0, 0, 1]).T)

    # 所有星点的天球坐标系下的坐标
    allStar = catalog[:, 1: 4]

    # 所有星点方向与视轴方向的夹角
    allDist = np.arccos(allStar.dot(S))

    # 将天球坐标系转换到星敏感器坐标系
    allStar = Rbc.dot(allStar.T)

    # 过滤出投影在图像中的星点并保存其相关信息
    cnt = 0
    stars = np.zeros((500, 7))
    for i in range(catalog.shape[0]):
        if allDist[i] < 0.75 * fov * np.pi / 180:
            star = allStar[:, i]
            x = - f * star[0] / star[2] / dx + cx
            y = - f * star[1] / star[2] / dy + cy
            if x > 0 and x < h and y > 0 and y < w:
                stars[cnt, :5] = catalog[i, :5]
                stars[cnt, 5:] = [x, y]
                cnt += 1
    stars = stars[:cnt, :]
  
    # 建立图像
    resImg = np.zeros((h, w))
    for star in stars:
        put_stars(resImg, star[6], star[5], 10000 / pow(2.51, star[4] - 2), delta = 1.0, winvisible = False, winradius = 10)

    # 添加噪声
    resImg += np.random.randn(h, w) * 3 + 10

    # 阈值截断
    resImg[np.where(resImg > 255)] = 255
    resImg[np.where(resImg < 0)] = 0

    # 返回图像
    return resImg

def rotate(yaw, pitch, roll):
    sina = np.sin(pitch)
    cosa = np.cos(pitch)
    sinb = np.sin(yaw)
    cosb = np.cos(yaw)
    sinc = np.sin(roll)
    cosc = np.cos(roll)

    # Rotation matrix
    Rx = np.array([[1, 0, 0], [0, cosa, sina], [0, -sina, cosa]])
    Ry = np.array([[cosb, 0, -sinb], [0, 1, 0], [sinb, 0, cosb]])
    Rz = np.array([[cosc, sinc, 0], [-sinc, cosc, 0], [0, 0, 1]])
    R = (Rx.dot(Ry).dot(Rz)).T
    return R

def genDynamic(attitude, attSpd, expo):

    # 读星库
    catalog = np.loadtxt('./params/sao60.txt', dtype = float)

    # 各项参数
    h, w = 2048, 2048
    cx, cy, dx, dy, f = [h / 2, w / 2, 0.0055, 0.0055, 25.0]
    fov = np.arctan((cx * dx) / f) * 180 / np.pi * 2

    # 角度转换为弧度制
    ra, dec, rol = np.array(attitude) * np.pi / 180

    # 姿态转换矩阵：天球坐标系 -> 星敏感器坐标系
    r11 = - np.cos(rol) * np.sin(ra) - np.sin(rol) * np.sin(dec) * np.cos(ra)
    r12 = np.cos(rol) * np.cos(ra) - np.sin(rol) * np.sin(dec) * np.sin(ra)
    r13 = np.sin(rol) * np.cos(dec)
    r21 = np.sin(rol) * np.sin(ra) - np.cos(rol) * np.sin(dec) * np.cos(ra)
    r22 = - np.sin(rol) * np.cos(ra) - np.cos(rol) * np.sin(dec) * np.sin(ra)
    r23 = np.cos(rol) * np.cos(dec)
    r31 = np.cos(dec) * np.cos(ra)
    r32 = np.cos(dec) * np.sin(ra)
    r33 = np.sin(dec)

    R = np.array([[r11, r12, r13], 
                  [r21, r22, r23],
                  [r31, r32, r33]])
    # 建立图像
    resImg = np.zeros((h, w))
    for i in range(expo):

        yaw, pitch, roll = (np.array(attSpd) * np.pi / 180) * i / 1000
        Rs = rotate(yaw, pitch, roll)
        # 姿态转换矩阵：星敏感器坐标系 -> 天球坐标系
        Rbc = Rs.dot(R)
        Rcb = Rbc.T

        # 视轴指向
        S = Rcb.dot(np.array([0, 0, 1]).T)

        # 所有星点的天球坐标系下的坐标
        allStar = catalog[:, 1: 4]

        # 所有星点方向与视轴方向的夹角
        allDist = np.arccos(allStar.dot(S))

        # 将天球坐标系转换到星敏感器坐标系
        allStar = Rbc.dot(allStar.T)

        # 过滤出投影在图像中的星点并保存其相关信息
        cnt = 0
        stars = np.zeros((500, 7))
        for i in range(catalog.shape[0]):
            if allDist[i] < 0.75 * fov * np.pi / 180:
                star = allStar[:, i]
                x = - f * star[0] / star[2] / dx + cx
                y = - f * star[1] / star[2] / dy + cy
                if x > 0 and x < h and y > 0 and y < w:
                    stars[cnt, :5] = catalog[i, :5]
                    stars[cnt, 5:] = [x, y]
                    cnt += 1
        stars = stars[:cnt, :]
    

        for star in stars:
            put_stars(resImg, star[6], star[5], 120 / pow(2.51, star[4] - 2), delta = 1.0, winvisible = False, winradius = 10)
    # 添加噪声
    resImg += np.random.randn(h, w) * 3 + 10

    # 阈值截断
    resImg[np.where(resImg > 255)] = 255
    resImg[np.where(resImg < 0)] = 0

    # 返回图像
    return resImg

if __name__ == "__main__":

    # attitude = [313.695954319231, 25.7233297886105, 115.765983323676]
    attitude = [12.0, 24.0, 36.0]
    attspd = [5, 5, 10]
    params = [1024, 1024, 0.0055, 0.0055, 25]

    res = genDynamic(attitude, attspd, 100)
    # plt.figure()
    # plt.imshow(res, cmap='gray', vmin=0, vmax=255)
    # plt.show()
    img = Image.fromarray(res)
    img = img.convert('L')
    img.save('./graph/test.png')

  
