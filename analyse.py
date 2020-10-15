import numpy as np
import cv2
import angest.est as ae

def get_mse(real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    return sum([(x - real) ** 2 for x in records_predict]) / len(records_predict)

data = np.load('save20noise.npy')
mse = get_mse(0, np.sort(data).tolist())

print(np.sort(np.abs(data)))
print('MSE:', mse)
print('R>0.2:', len(np.where(np.abs(data) > 0.2)[0]))
print('R>0.4:',len(np.where(np.abs(data) > 0.4)[0]))
print('R>0.6:',len(np.where(np.abs(data) > 0.6)[0]))
print('R>1.0:',len(np.where(np.abs(data) > 1.0)[0]))

# src = cv2.imread(r'./graph/dynamic/variable_noise/0/-31_161_0_30.00001156757613.png', 0)
# print(src)
# # blured = src
# src = cv2.blur(src, (3, 3))
# direction =  ae.Direction_estimate(src)
# print(direction)
