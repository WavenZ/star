import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class StarGenerator(object):
    
    def __init__(self, filename, figsize = (2048, 2048)):
        '''
            初始化
        '''
        self.filename = filename
        self.figsize = figsize
        self.stars = self.parse_catalogue()
        self.K = np.array([[5800, 0, figsize[0] // 2], 
                           [0, 5800, figsize[1] // 2], 
                           [0, 0, 1]])


    def generate(self, pitch, yaw, roll, starsize = 1.3, winvisible = False, winradius = 50):
        '''
            生成星图
        '''
        self.pitch = self.to_rad(pitch)
        self.yaw = self.to_rad(yaw)
        self.roll = self.to_rad(roll)
        
        # 三角运算
        sina = np.sin(self.yaw - np.pi / 2)
        cosa = np.cos(self.yaw - np.pi / 2)
        sinb = np.sin(self.pitch + np.pi / 2)
        cosb = np.cos(self.pitch + np.pi / 2)
        sinc = np.sin(self.roll) 
        cosc = np.cos(self.roll)
        
        # 视轴指向
        rx = np.cos(self.pitch) * np.cos(self.yaw)
        ry = np.cos(self.pitch) * np.sin(self.yaw)
        rz = np.sin(self.pitch)
        
        # Rotation matrix
        Rx = np.array([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]])
        Ry = np.array([[1, 0, 0], [0, cosb, -sinb], [0, sinb, cosb]])
        Rz = np.array([[cosc, -sinc, 0], [sinc, cosc, 0], [0, 0, 1]])
        R = (Rx.dot(Ry).dot(Rz)).T
        # 转换为天球坐标系
        X = np.array([np.cos(self.stars[:, 3]) * np.cos(self.stars[:, 2]),
                      np.cos(self.stars[:, 3]) * np.sin(self.stars[:, 2]),
                      np.sin(self.stars[:, 3])])
        
        # 筛选出星敏正面的星点
        num = self.stars.shape[0]
        starID = np.linspace(0, num, num, endpoint = False)
        starID = starID[np.where(np.inner(X.T, [rx, ry, rz]) > 0)]
        X = X.T[np.where(np.inner(X.T, [rx, ry, rz]) > 0)].T
        
        # 将星点投影到图像平面 
        x = self.K.dot(R).dot(X).reshape(3, -1)
        y = np.array([(x[1] / x[2]).astype(int), (x[0] / x[2]).astype(int)])
        
        # 筛选出落在视场内的星点
        starID = starID[np.where((y[0] >= 0) & (y[0] < 2048) & (y[1] >= 0) & (y[1] < 2048))]
        y = y.T[np.where((y[0] >= 0) & (y[0] < 2048) & (y[1] >= 0) & (y[1] < 2048))].T
        

        
        # 在星图中绘制星点
        for i in range(y.shape[1]):
            self.put_stars(img, y[0, i], y[1, i], 10000 / pow(2.51, self.stars[int(starID[i]), 1] - 2), 
                           starsize, winvisible, winradius)
        
        # 添加噪声
        self.add_noise(img, 3)
        
        # 灰度截取
        img[np.where(img > 255)] = 255
        img[np.where(img < 0)] = 0
        
        return img, y.shape[1]

    def add_noise(self, img, sigma):
        '''
            添加噪声
        '''
        h, w = img.shape
        noise = np.random.randn(h, w) * sigma
        img += noise

    def to_rad(self, angle):
        '''
            角度转弧度
        '''
        return angle / 180 * np.pi
    
    def parse_catalogue(self):
        '''
            读星库
        '''
        return np.loadtxt(self.filename, dtype = float)[:, :4]

    def gaussian(self, E, delta, x, y, x0, y0):
        '''
            二维高斯函数
        '''
        return E / (2 * np.pi * delta ** 2) * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * delta ** 2))

    def put_stars(self, img, x0, y0, E, delta = 1.3, winvisible = False, winradius = 50):
        '''
            添加星点
        '''
        up = x0 - winradius if x0 - winradius >= 0 else 0
        down = x0 + winradius + 1 if x0 + winradius + 1 <= img.shape[0] else img.shape[0]
        left = y0 - winradius if y0 - winradius >= 0 else 0
        right = y0 + winradius + 1 if y0 + winradius + 1 <= img.shape[1] else img.shape[1]
        x = np.linspace(up, down - 1, down - up)
        y = np.linspace(left, right - 1, right - left)
        X, Y = np.meshgrid(x, y)
        X = X.T
        Y = Y.T
        if winvisible is True:
            img[up : down, left : right] += 50
        img[up : down, left : right] += E / (2 * np.pi * delta ** 2) * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * delta ** 2))


if __name__ == "__main__":
    G = StarGenerator('sao60')
    pitch, yaw, roll = 0, 0, 0
    img, starnum = G.generate(pitch, yaw, roll, winvisible = False)
    print('Image size: {}'.format(img.shape))
    print('Total stars: {}'.format(starnum))
    print('Pitch: {}, Yaw: {}, Roll: {}'.format(pitch, yaw, roll))

    # 绘图
    plt.figure(figsize = (5, 5))
    plt.subplot(111, facecolor = 'k')
    plt.xlim([0, 2048])
    plt.ylim([0, 2048])
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值
    plt.imsave('{}_{}_{}.png'.format(pitch, yaw, roll), img, cmap = 'gray')
    plt.imshow(img[::-1], cmap = 'gray', vmin = 0, vmax = 150)
    plt.show()