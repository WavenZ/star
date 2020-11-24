import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class StarGenerator(object):

    def __init__(self, filename, figsize = (2048, 2048)):
        '''Initialization.'''

        self.filename = filename
        self.figsize = figsize
        self.stars = self.parse_catalogue()
        self.K = np.array([[5800, 0, figsize[0] // 2],
                           [0, 5800, figsize[1] // 2],
                           [0, 0, 1]])

    def rotate(self, pitch, yaw, roll):
        sina = np.sin(yaw - np.pi / 2)
        cosa = np.cos(yaw - np.pi / 2)
        sinb = np.sin(pitch + np.pi / 2)
        cosb = np.cos(pitch + np.pi / 2)
        sinc = np.sin(roll)
        cosc = np.cos(roll)

        # Rotation matrix
        Rx = np.array([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]])
        Ry = np.array([[1, 0, 0], [0, cosb, -sinb], [0, sinb, cosb]])
        Rz = np.array([[cosc, -sinc, 0], [sinc, cosc, 0], [0, 0, 1]])
        R = (Rx.dot(Ry).dot(Rz)).T

        return R

    def generate(self, pitch, yaw, roll, hspd, vspd, exposure = 100, 
                 starsize = 1.2, winvisible = True, winradius = 10, noise = 3):
        '''Generate star image.'''

        img = np.zeros(self.figsize)

        origin_pitch = self.to_rad(pitch)
        origin_yaw = self.to_rad(yaw)
        origin_roll = self.to_rad(roll)

        R = self.rotate(origin_pitch, origin_yaw, origin_roll)

        # Optical axis direction.
        rx = np.cos(origin_pitch) * np.cos(origin_yaw)
        ry = np.cos(origin_pitch) * np.sin(origin_yaw)
        rz = np.sin(origin_pitch)

        # Converto to celestial sphere.
        X = np.array([np.cos(self.stars[:, 3]) * np.cos(self.stars[:, 2]),
                np.cos(self.stars[:, 3]) * np.sin(self.stars[:, 2]),
                np.sin(self.stars[:, 3])])

        # Screen out the front stars.
        num = self.stars.shape[0]
        starID = np.linspace(0, num, num, endpoint = False)
        starID = starID[np.where(np.inner(X.T, [rx, ry, rz]) > 0)]
        X = X.T[np.where(np.inner(X.T, [rx, ry, rz]) > 0)].T

        # Project the star to the iamge plane.
        x = self.K.dot(R).dot(X).reshape(3, -1)
        y = np.array([(x[1] / x[2]), (x[0] / x[2])])


        for i in range(exposure):

            y[0, :] += vspd
            y[1, :] += hspd

            # Select the stars in the field of view.
            starID = starID[np.where((y[0] >= 0) & (y[0] < 2048) & (y[1] >= 0) & (y[1] < 2048))]
            y = y.T[np.where((y[0] >= 0) & (y[0] < 2048) & (y[1] >= 0) & (y[1] < 2048))].T
            # Place stars in image.
            for i in range(y.shape[1]):
                self.put_stars(img, y[0, i], y[1, i], 3000 / pow(2.51, self.stars[int(starID[i]), 1] - 2) / 30,
                            starsize, winvisible, winradius)


        
        # Add noise.
        img[np.where(img > 255)] = 255
        img[np.where(img < 0)] = 0
        self.add_noise(img, noise)

        # Grayscale inerception.
        img[np.where(img > 255)] = 255
        img[np.where(img < 0)] = 0

        return img, y.shape[1]

    def add_noise(self, img, sigma):
        '''Add some noise'''

        h, w = img.shape
        img += np.random.randn(h, w) * sigma + 10

    def to_rad(self, angle):
        '''Angle to radian.'''

        return angle / 180 * np.pi

    def parse_catalogue(self):
        '''Parse the specified catalogue.'''
        data = np.loadtxt(self.filename, dtype = float)[:, :4]
        data = data[np.where(data[:, 1] >= 1.0)]
        return data

    def gaussian(self, E, delta, x, y, x0, y0):
        '''2-D Gaussian function.'''

        return E / (2 * np.pi * delta ** 2) * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * delta ** 2))

    def put_stars(self, img, x0, y0, E, delta = 1.3, winvisible = True, winradius = 50):
        '''Place stars to the specified image.'''

        up = int(x0) - winradius if int(x0) - winradius >= 0 else 0
        down = int(x0) + winradius + 1 if int(x0) + winradius + 1 <= img.shape[0] else img.shape[0]
        left = int(y0) - winradius if int(y0) - winradius >= 0 else 0
        right = int(y0) + winradius + 1 if int(y0) + winradius + 1 <= img.shape[1] else img.shape[1]
        x = np.linspace(up, down - 1, down - up)
        y = np.linspace(left, right - 1, right - left)
        X, Y = np.meshgrid(x, y)
        X, Y = X.T, Y.T
        if winvisible is True:
            print(winvisible)
            img[up : down, left : right] += 1
        img[up : down, left : right] += E / (2 * np.pi * delta ** 2) * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * delta ** 2))


if __name__ == "__main__":
    # Initialization.
    G = StarGenerator('sao60')
    
    # Set the speed.
    hspd, vspd = 0.1, 0.1
    
    # Generate.
    for i in range(100):
        print('[GEN ID]:', i)
        t1 = time.time()
        # Randomly generate the initialization position.
        pitch = np.random.randint(-90, 90)
        yaw = np.random.randint(0, 360)
        roll = 0
        image, _ = G.generate(pitch, yaw, roll, hspd, vspd, exposure = 100, winvisible = False, noise = 3)
        # plt.figure()
        # plt.imshow(image, cmap='gray', vmin = 0, vmax = 255)
        # plt.show()
        # plt.imsave('../graph/dynamic/fix45/{}.png'.format(i), image, cmap = 'gray', vmin = 0, vmax = 255)
        t2 = time.time()
        im = Image.fromarray(image)
        im = im.convert('L')
        im.save('../graph/dynamic/fix/{}.png'.format(i))
        # plt.show()    
        t3 = time.time()
        print(t2 - t1, t3 - t2)
    # plt.figure(figsize = (5, 5))
    # plt.subplot(111, facecolor = 'k')
    # plt.xlim([0, 2048])
    # plt.ylim([0, 2048])
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(img[::-1], cmap = 'gray', vmin = 0, vmax = 150)
    # plt.show()