import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


# import generator.gen_static_new as gsn

from cv2 import cv2
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
if __name__ == "__main__":
    # attitude = [30, 45, 60]
    attitude = [0, 0, 0]
    spd = [30, 0, 0]
    # 角度转换为弧度制
    ra, dec, rol = np.array(attitude) * np.pi / 180
    ras, decs, rols = np.array(spd) * np.pi / 180

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
    
    # 姿态转换矩阵：天球坐标系 -> 星敏感器坐标系
    s11 = - np.cos(rols) * np.sin(ras) - np.sin(rols) * np.sin(decs) * np.cos(ras)
    s12 = np.cos(rols) * np.cos(ras) - np.sin(rols) * np.sin(decs) * np.sin(ras)
    s13 = np.sin(rols) * np.cos(decs)
    s21 = np.sin(rols) * np.sin(ras) - np.cos(rols) * np.sin(decs) * np.cos(ras)
    s22 = - np.sin(rols) * np.cos(ras) - np.cos(rols) * np.sin(decs) * np.sin(ras)
    s23 = np.cos(rols) * np.cos(decs)
    s31 = np.cos(decs) * np.cos(ras)
    s32 = np.cos(decs) * np.sin(ras)
    s33 = np.sin(decs)

    R12 = np.array([[s11, s12, s13], 
                    [s21, s22, s23],
                    [s31, s32, s33]])



    # 姿态转换矩阵：星敏感器坐标系 -> 天球坐标系
    Rcb = Rbc.T
    print(Rbc)
    curr = np.array([0, 0, 1]).T
    yaw, pitch, roll = np.array([0, 0, 30]) * np.pi / 180
    R = rotate(yaw, pitch, roll)
    print(R.dot(Rbc).dot(curr))
    print(R.dot(Rbc.dot(curr)))