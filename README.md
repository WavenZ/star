### 高动态星点提取方法
#### python版本
  - [x] 3.6.0 - 3.8.0

#### 已有功能
  - [x] 静态星点仿真
  - [x] 动态星点仿真
  - [x] 星点运动方向相同时的方向估计
  - [x] 星点运动方向不同时的方向估计

#### 当前工作
  - [x] 合并上述后两项工作 


#### 依赖库
 - numpy、maplotlib、opencv、sklearn、scipy、PIL

#### 目录对应说明
 - estimate: 高动态星图运动方向估计程序
 - generator: 动静态星图仿真生成程序
 - graph: 星图文件夹

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
3. 动态仿真连续多帧星图生成

