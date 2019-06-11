## 目标：

本章节你需要学习以下内容:
- 我们将学习如何从立体图像创建深度图。

## 基础

在上一节中我们学习了对极约束的基本概念和相关术语。如果同一场景有两幅图像的话我们在直觉上就可以获得图像的深度信息。下面是的这幅图和其中的数学公式证明我们的直觉是对的。（图像来源 image courtesy）
![image9](https://docs.opencv.org/4.0.0/stereo_depth.jpg)

上图包含等效三角形。编写等效方程将产生以下结果：

$$disparity = x - x' = \frac{Bf}{Z}$$

x 和 x' 分别是图像中的点到 3D 空间中的点和到摄像机中心的距离。B 是这两个摄像机之间的距离，f 是摄像机的焦距。上边的等式告诉我们点的深度与x 和 x' 的差成反比。所以根据这个等式我们就可以得到图像中所有点的深度图。

这样就可以找到两幅图像中的匹配点了。前面我们已经知道了对极约束可以使这个操作更快更准。一旦找到了匹配，就可以计算出 disparity 了。让我们看看在 OpenCV 中怎样做吧。

## 代码实现
下面的代码片段显示了创建视差图的简单过程。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('tsukuba_l.png',0)
imgR = cv.imread('tsukuba_r.png',0)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
```

下图包含原始图像（左）及其视差图（右）。如图所见，结果受到高度噪音的污染。通过调整numDisparities和blockSize的值，你可以获得更好的结果。

![image10](https://docs.opencv.org/4.0.0/disparity_map.jpg)