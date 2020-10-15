import cv2;
import numpy as np
from cv2 import cv2
from PIL import Image

src = cv2.imread('./graph/100ms_adg6.bmp', 0)
src = 255 - src
im = Image.fromarray(src)
im = im.convert('L')
im.save('./graph/100ms_adg6_reverse.png')

