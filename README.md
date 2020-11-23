### 高动态星点提取方法
#### python版本
  - [x] 3.6.0 - 3.8.0

#### 已有功能
  - [x] 静态星点仿真
  - [x] 动态星点仿真
  - [x] 估计星点的旋转中心（平行时为无穷远处）
  - [x] 图像增强、星点修复
  - [x] 姿态解算、重投影

#### 当前工作
  - [x] 解决重投影误差较大的问题 


#### 依赖库
 - numpy、maplotlib、opencv、sklearn、scipy、PIL

#### 目录对应说明
 - estimate: 高动态星图运动方向估计程序
 - generator: 动静态星图仿真生成程序
 - extract: 星点提取程序
 - identifier: 星图识别、姿态解算、重投影程序
 - params: 一些参数或者星库文件
 - tools: 一些工具程序
 - graph: 星图文件夹
 - temp: 临时文件夹，舍不得删的放这里

#### 项目使用示例

1. 静态仿真星图生成
```cpp
import matplotlib.pyplot as plt
import generator.gen_static as gs

if __name__ == "__main__":

    G = gs.StarGenerator('sao60')
    pitch, yaw, roll = 10, 30, 60
    img, starnum = G.generate(pitch, yaw, roll, winvisible = False)
    print('[starnum]:', starnum)
    plt.figure()
    plt.imshow(img, cmap='gray', )
    plt.show()
```
2. 动态仿真星图生成
```python
import matplotlib.pyplot as plt
import generator.gen_dynamic as gd

if __name__ == "__main__":

    G = gd.StarGenerator('sao60')
    pitch, yaw, roll = 10, 30, 60
    pitchspd, yawspd, rollspd = 0, 5, 0
    img, starnum = G.generate(pitch, yaw, roll, pitchspd, yawspd, rollspd, winvisible = False)
    print('[starnum]:', starnum)
    plt.figure()
    plt.imshow(img, cmap='gray', )
    plt.show()

```
3. 旋转中心估计
```python
import cv2
from cv2 import cv2

if __name__ == "__main__":

  src = cv2.imread('./graph/5_1.png')
  src = cv2.blur(src, (3, 3))
  rot_center = ae.Direction_estimate(src)

```
4. 星点提取、质心定位
```python
import cv2
from cv2 import cv2

if __name__ == "__main__":

  src = cv2.imread('./graph/5_1.png')
  src = cv2.blur(src, (3, 3))
  
  # 估计旋转中心
  rot_center = ae.Direction_estimate(src)
  
  # 星点提取、质心定位
  retImg, centers, cnt = ex.extract(src.copy(), rot_center)
  centers = centers[:cnt]
```
5. 星图识别、姿态解算、重投影
```python
import cv2
from cv2 import cv2

if __name__ == "__main__":

  src = cv2.imread('./graph/5_1.png')
  src = cv2.blur(src, (3, 3))
  
  # 估计旋转中心
  rot_center = ae.Direction_estimate(src)
  
  # 星点提取、质心定位
  retImg, centers, cnt = ex.extract(src.copy(), rot_center)
  centers = centers[:cnt]
  
  # 星图识别、姿态解算
  res = ip.identify(centers)[-1] * 180 / np.pi
  
  # 重投影
  retImg = ip.reProjection(retImg, res, [1024, 1024, 0.0055, 0.0055, 25.5])
  plt.figure()
  plt.imshow(retImg, cmap='gray', vmin=0, vmax=255)
  plt.show()
```

