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
        X = X.T
        Y = Y.T
        if winvisible is True:
            img[up : down, left : right] += 50
        img[up : down, left : right] += E / (2 * np.pi * delta ** 2) * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * delta ** 2))

def genStatic(attitude):

    # 读星库
    catalog = np.loadtxt('./params/sao60.txt', dtype = float)

    # 各项参数
    h, w = 2048, 2048
    cx, cy, dx, dy, fov = [h / 2, w / 2, 0.0055, 0.0055, 25.5]
    f = (cx * dx) / np.tan(fov / 2 * np.pi / 180)

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

def reProjection(src, attitude, params):

    # 读星库
    catalog = np.loadtxt('./params/sao60.txt', dtype = float)

    # 各项参数
    h, w = 2048, 2048
    cx, cy, dx, dy, fov = params
    f = (cx * dx) / np.tan(fov / 2 * np.pi / 180)

    # 角度转换为弧度制
    pitch, yaw, roll = np.array(attitude) * np.pi / 180

    # 姿态转换矩阵：天球坐标系 -> 星敏感器坐标系
    r11 = - np.cos(roll) * np.sin(pitch) - np.sin(roll) * np.sin(yaw) * np.cos(pitch)
    r12 = np.cos(roll) * np.cos(pitch) - np.sin(roll) * np.sin(yaw) * np.sin(pitch)
    r13 = np.sin(roll) * np.cos(yaw)
    r21 = np.sin(roll) * np.sin(pitch) - np.cos(roll) * np.sin(yaw) * np.cos(pitch)
    r22 = - np.sin(roll) * np.cos(pitch) - np.cos(roll) * np.sin(yaw) * np.sin(pitch)
    r23 = np.cos(roll) * np.cos(yaw)
    r31 = np.cos(yaw) * np.cos(pitch)
    r32 = np.cos(yaw) * np.sin(pitch)
    r33 = np.sin(yaw)

    R = np.array([[r11, r12, r13], 
                    [r21, r22, r23], 
                    [r31, r32, r33]])
    
    # 视轴指向
    S = R.T.dot(np.array([0, 0, 1]).T)

    # 所有星点的天球坐标系下的坐标
    allStar = catalog[:, 1: 4]

    # 所有星点方向与视轴方向的夹角
    allDist = np.arccos(allStar.dot(S))

    # 将天球坐标系转换到星敏感器坐标系
    allStar = R.dot(allStar.T)

    # 过滤出投影在图像中的星点并保存其相关信息
    cnt = 0
    stars = np.zeros((500, 7))
    for i in range(catalog.shape[0]):
        # 过滤与视轴夹角较大的恒星
        if allDist[i] < 0.75 * fov * np.pi / 180:
            # 取出该星点
            star = allStar[:, i]
            # 计算投影在图像平面的坐标
            x = - f * (star[0] / star[2]) / dx + cx
            y = - f * (star[1] / star[2]) / dy + cy - 1024
            # 筛选落在图像中的星保存到 stars 中
            if x > 0 and x < src.shape[1] and y > 0 and y < src.shape[0]:
                stars[cnt, :5] = catalog[i, :5]
                stars[cnt, 5:] = [x, y]
                cnt += 1
    stars = stars[:cnt, :]
  
    # 建立图像
    resImg = Image.fromarray(src).convert('RGB')
    font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf', 20)
    anno = ImageDraw.Draw(resImg)
    
    # 画星点和标注星等信息
    for star in stars:
        anno.ellipse((star[5] - 4, star[6] - 4, star[5] + 4, star[6] + 4), fill = 'red')
        anno.text((star[5] + 10, star[6] + 10), '{:.2f}'.format(star[4]), font = font, fill = 'red')
        
    resImg = np.array(resImg)

    # 返回图像
    return resImg  

if __name__ == "__main__":

    attitude = [313.695954319231, 25.7233297886105, 115.765983323676]
    params = [1024, 1024, 0.0055, 0.0055, 25]

    res = genStatic(attitude)
    

    # src1 = cv2.imread('./graph/5_2_res.png', 0)
    # res1 = reProjection(src1, attitude, params)
    # res2 = genStatic(attitude)
    # plt.figure()
    # plt.imshow(res1, cmap='gray', vmin=0, vmax=255)
    # plt.figure()
    # plt.imshow(res2, cmap='gray', vmin=0, vmax=255)
    # Save
    img = Image.fromarray(res[1024:, :])
    img = img.convert('L')
    img.save('./graph/test.png')
