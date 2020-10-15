import random
from PIL import Image
import numpy as np
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

    def generate(self, pitch, yaw, roll, pitchspd, yawspd, rollspd, exposure = 100, 
                 starsize = 1.3, winvisible = False, winradius = 50, noise = 3):
        '''Generate star image.'''

        img = np.zeros(self.figsize)


        origin_pitch = self.to_rad(pitch)
        origin_yaw = self.to_rad(yaw)
        origin_roll = self.to_rad(roll)

        # curr_pitch = self.to_rad(pitch)
        # curr_yaw = self.to_rad(yaw)
        # curr_roll = self.to_rad(roll)

        delta_pitch = 0
        delta_yaw = 0
        delta_roll = 0

        
        for i in range(exposure):
            delta_pitch += self.to_rad(pitchspd / 1000)
            delta_yaw += self.to_rad(yawspd / 1000)
            delta_roll += self.to_rad(rollspd / 1000)
            # print(delta_pitch, delta_yaw, delta_roll)

            R = self.rotate(origin_pitch, origin_yaw, origin_roll)
            delta_R = self.rotate(delta_pitch, delta_yaw, delta_roll)
            # print(delta_R)
            R = delta_R.dot(R)

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

            # Select the stars in the field of view.
            starID = starID[np.where((y[0] >= 0) & (y[0] < 2048) & (y[1] >= 0) & (y[1] < 2048))]
            y = y.T[np.where((y[0] >= 0) & (y[0] < 2048) & (y[1] >= 0) & (y[1] < 2048))].T

            # Place stars in image.
            for i in range(y.shape[1]):
                self.put_stars(img, y[0, i], y[1, i], 10000 / pow(2.51, self.stars[int(starID[i]), 1] - 2) / 30,
                            starsize, winvisible, winradius)

        # Add noise.
        img[np.where(img > 255)] = 255
        img[np.where(img < 0)] = 0
        self.add_noise(img, noise)

        # Grayscale inerception.
        img[np.where(img > 255)] = 255
        img[np.where(img < 0)] = 0
        return img, y.shape[1]

    def generateMulti(self, pitch, yaw, roll, pitchspd_max, yawspd_max, rollspd_max, 
                        pitchspd_acc, yawspd_acc, rollspd_acc, exposure = 100, 
                        starsize = 1.3, winvisible = False, winradius = 50, noise = 3, frames = 10):
        '''Gernertate continuous frames images.'''

        # Angle to raidan.
        origin_pitch = self.to_rad(pitch)
        origin_yaw = self.to_rad(yaw)
        origin_roll = self.to_rad(roll)

        # Optical axis direction.
        rx = np.cos(origin_pitch) * np.cos(origin_yaw)
        ry = np.cos(origin_pitch) * np.sin(origin_yaw)
        rz = np.sin(origin_pitch)

        # Original rotation matrix.
        R = self.rotate(origin_pitch, origin_yaw, origin_roll)

        # Convert to celestial sphere.
        X = np.array([np.cos(self.stars[:, 3]) * np.cos(self.stars[:, 2]),
                      np.cos(self.stars[:, 3]) * np.sin(self.stars[:, 2]),
                      np.sin(self.stars[:, 3])])

        # curr_pitch = self.to_rad(pitch)
        # curr_yaw = self.to_rad(yaw)
        # curr_roll = self.to_rad(roll)

        pitchspd, yawspd, rollspd = 0, 0, 0
        delta_pitch, delta_yaw, delta_roll = 0, 0, 0

        for k in range(frames):

            img = np.zeros(self.figsize)

            # The simulation granularity is 1ms
            for _ in range(exposure):

                # Update the angular velocity.
                if pitchspd < pitchspd_max:
                    pitchspd += (pitchspd_acc / 1000)
                if yawspd < yawspd_max:
                    yawspd += (yawspd_acc / 1000)
                if rollspd < rollspd_max:
                    rollspd += (rollspd_acc / 1000)

                # Update the attitude.
                delta_pitch += self.to_rad(pitchspd / 1000)
                delta_yaw += self.to_rad(yawspd / 1000)
                delta_roll += self.to_rad(rollspd / 1000)

                # Update the attitude matrix.
                delta_R = self.rotate(delta_pitch, delta_yaw, delta_roll)
                R = delta_R.dot(R)


                # Screen out the front stars.
                num = self.stars.shape[0]
                starID = np.linspace(0, num, num, endpoint = False)
                starID = starID[np.where(np.inner(X.T, [rx, ry, rz]) > 0)]
                X = X.T[np.where(np.inner(X.T, [rx, ry, rz]) > 0)].T

                # Project the star to the image plane.
                x = self.K.dot(R).dot(X).reshape(3, -1)
                y = np.array([(x[1] / x[2]), (x[0] / x[2])])

                # Select the stars in the field of view.
                starID = starID[np.where((y[0] >= 0) & (y[0] < 2048) & (y[1] >= 0) & (y[1] < 2048))]
                y = y.T[np.where((y[0] >= 0) & (y[0] < 2048) & (y[1] >= 0) & (y[1] < 2048))].T

                # print(y[1][5:10])
                # Place stars in image.
                for i in range(y.shape[1]):
                    self.put_stars(img, y[0, i], y[1, i], 10000 / pow(2.51, self.stars[int(starID[i]), 1] - 2) / 30,
                                starsize, winvisible, winradius)

            # Add some noise.
            img[np.where(img > 255)] = 255
            img[np.where(img < 0)] = 0
            self.add_noise(img, noise)

            # GrayScale interception.
            img[np.where(img > 255)] = 255
            img[np.where(img < 0)] = 0

            # Save image.
            plt.imsave('./graph/dynamic/multi_frame/b_0_5_10_120/{}.png'.format(k + 1), img, cmap = 'gray', vmin = 0, vmax = 255)

    def add_noise(self, img, sigma):
        '''Add some noise'''

        h, w = img.shape
        noise = np.random.randn(h, w) * sigma + 10
        img += noise

    def to_rad(self, angle):
        '''Angle to radian.'''

        return angle / 180 * np.pi

    def parse_catalogue(self):
        '''Parse the specified catalogue.'''

        return np.loadtxt(self.filename, dtype = float)[:, :4]

    def gaussian(self, E, delta, x, y, x0, y0):
        '''2-D Gaussian function.'''

        return E / (2 * np.pi * delta ** 2) * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * delta ** 2))

    def put_stars(self, img, x0, y0, E, delta = 1.3, winvisible = False, winradius = 50):
        '''Place stars to the specified image.'''

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


if __name__ == "__main__":
    # Initialization.
    G = StarGenerator('sao60')
    
    # Randomly generate the initialization position.
    pitch = np.random.randint(-90, 90)
    yaw = np.random.randint(0, 360)
    roll = 0
    
    # Set the acceleration and the final speed.
    pitchspd, yawspd, rollspd = 0, 5, 10
    pitchspd_acc, yawspd_acc, rollspd_acc = 0, 1, 2
    
    # Generate.
    G.generateMulti(pitch, yaw, roll, pitchspd, yawspd, rollspd, pitchspd_acc, yawspd_acc, rollspd_acc, exposure = 100, winvisible = False, noise = 3, frames = 120)


    # plt.figure(figsize = (5, 5))
    # plt.subplot(111, facecolor = 'k')
    # plt.xlim([0, 2048])
    # plt.ylim([0, 2048])
    # plt.xticks([])
    # plt.yticks([])
    # # plt.imsave('{}_{}_{}.png'.format(pitch, yaw, roll), img, cmap = 'gray')
    # plt.imshow(img[::-1], cmap = 'gray', vmin = 0, vmax = 150)
    # plt.show()