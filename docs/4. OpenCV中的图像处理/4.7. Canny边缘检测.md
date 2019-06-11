## 目标：
- Canny边缘检测的概念
- OpenCV的功能：`cv.Canny()`

## 理论
Canny边缘检测是一种流行的边缘检测算法，它是由John F. Canny开发的。

#### 1. 这是一个多阶段算法，我们将了解其中的每个阶段。
#### 2. 降噪
由于边缘检测易受图像中的噪声影响，因此第一步是使用5x5高斯滤波器去除图像中的噪声。我们在之前的章节中已经看到了这一点。

#### 3. 计算图像的强度梯度

然后在水平和垂直方向上用Sobel核对平滑后的图像进行滤波，以获得水平方向($$ G_{x} $$)和垂直方向($$ G_{y} $$)的一阶导数。从这两个图像中，我们可以找到每个像素的边缘梯度和方向，如下所示：

$$ Edge\_Gradient\left ( G \right )= \sqrt{G_{x}^{2}+G_{y}^{2}} $$

$$ Angle\left ( \theta  \right )= tan^{-1}\left ( \frac{G_{y}}{G_{x}} \right ) $$

渐变方向始终垂直于边缘。梯度方向被归为四类：垂直，水平，和两个对角线。

#### 4. 非极大值抑制

在获得梯度的大小和方向之后，完成图像的全扫描以去除可能不构成边缘的任何不需要的像素。为此，在每个像素处，检查像素是否是其在梯度方向上的邻域中的局部最大值。检查下图：

![image24](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image24.png)

A点位于边缘（垂直方向）。渐变方向与边缘垂直。B点和C点处于梯度方向。因此，用点B和C检查点A，看它是否形成局部最大值。如果是这样，则考虑下一阶段，否则，它被抑制（置零）。

简而言之，你得到的结果是具有“细边”的二进制图像。

#### 5. 滞后阈值

这个阶段决定哪些边缘都是边缘，哪些边缘不是边缘。为此，我们需要两个阈值，minVal和maxVal。强度梯度大于maxVal的任何边缘肯定是边缘，而minVal以下的边缘肯定是非边缘的，因此被丢弃。位于这两个阈值之间的人是基于其连通性的分类边缘或非边缘。如果它们连接到“可靠边缘”像素，则它们被视为边缘的一部分。否则，他们也被丢弃。见下图：
![image25](https://docs.opencv.org/4.0.0/nms.jpg)
边缘A高于maxVal，因此被视为“确定边缘”。虽然边C低于maxVal，但它连接到边A，因此也被视为有效边，我们得到完整的曲线。但边缘B虽然高于minVal并且与边缘C的区域相同，但它没有连接到任何“可靠边缘”，因此被丢弃。所以我们必须相应地选择minVal和maxVal才能获得正确的结果。
假设边是长线，这个阶段也会消除小像素噪声。
所以我们最终得到的是图像中的强边缘。

## OpenCV中的Canny边缘检测

OpenCV将以上所有步骤放在单个函数`cv.Canny()`中。我们将看到如何使用它。第一个参数是我们的输入图像。第二个和第三个参数分别是我们的minVal和maxVal。第三个参数是aperture_size,它是用于查找图像渐变的Sobel卷积核的大小。默认情况下它是3。最后一个参数是L2gradient，它指定用于查找梯度幅度的等式。如果它是True，它使用上面提到的更准确的等式，否则它使用这个函数：$$ Edge\_Gradient\left ( G \right )= \left | G_{x} \right |+\left | G_{y} \right | $$。默认情况下，它为False。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('messi5.jpg',0)
edges = cv.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
```

窗口将如下图显示：
![image26](https://docs.opencv.org/4.0.0/canny1.jpg)
