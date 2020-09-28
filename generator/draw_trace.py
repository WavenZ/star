import numpy as np
import matplotlib.pyplot as plt

# 旋转轴
alpha = np.pi / 3
beta = 90 / 180 * np.pi
rx = np.cos(beta) * np.cos(alpha)
ry = np.cos(beta) * np.sin(alpha)
rz = np.sin(beta)
# 旋转调度
theta = np.linspace(0, 1 * 2 * np.pi, 10001)
cost = np.cos(theta)
sint = np.sin(theta)
# 旋转矩阵
R = np.array([[rx * rx * (1 - cost) + cost, rx * ry * (1 - cost) - rz * sint, rx * rz * (1 - cost) + ry * sint],
              [rx * ry * (1 - cost) + rz * sint, ry * ry * (1 - cost) + cost, ry * rz * (1 - cost) - rx * sint],
              [rx * rz * (1 - cost) - ry * sint, ry * rz * (1 - cost) + rx * sint, rz * rz * (1 - cost) + cost]]).transpose(2, 0, 1)

# 摄像机矩阵
K = np.array([[5800, 0, 1024], 
              [0,5800, 1024], 
              [0, 0, 1]])
# 星光矢量X1
alpha = np.pi / 3
beta = 80 / 180 * np.pi
X1 = np.array([np.cos(beta) * np.cos(alpha),np.cos(beta) * np.sin(alpha),np.sin(beta)])
print(X1)
# 星光矢量X2
alpha = np.pi / 4
beta = -81 / 180 * np.pi
X2 = np.array([np.cos(beta) * np.cos(alpha),np.cos(beta) * np.sin(alpha),np.sin(beta)])
print(X2.shape)

img = np.zeros((2048, 2048))



# 投影
x1 = K.dot(R).dot(X1).reshape(3, 10001)
x2 = K.dot(R).dot(X2).reshape(3, 10001)

print(x1.shape)

img[(x1[0] / x1[2]).astype(np.int), (x1[1] / x1[2]).astype(np.int)] = 255
img[(x2[0] / x2[2]).astype(np.int), (x2[1] / x2[2]).astype(np.int)] = 255





# 绘图
plt.figure(figsize = (5, 5))
plt.subplot(111, facecolor = 'k')
plt.xlim([0, 2048])
plt.ylim([0, 2048])
plt.xticks([])  #去掉横坐标值
plt.yticks([])  #去掉纵坐标值
# plt.scatter(x1[0] / x1[2], x1[1] / x1[2], color = 'w', marker = 's')
# plt.scatter(x2[0] / x2[2], x2[1] / x2[2], color = 'w', marker = 's')
plt.imshow(img, cmap = 'gray')


plt.show()