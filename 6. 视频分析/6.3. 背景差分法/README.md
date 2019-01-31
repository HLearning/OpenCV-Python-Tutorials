## 目标：
- 熟悉OpenCV中可用的背景差分法。

## 基础
在许多基于视觉的应用程序中，背景减法是一个主要的预处理步骤。例如顾客统计，使用静态摄像机获取进入或离开房间的访客的数量，或者提取有关车辆等的信息的交通摄像机。在例子中，首先要将人或车单独提取出来。技术上来说，需要从静止的背景中提取移动的前景。

如果你有一张背景图像，比如没有顾客的房间，没有交通工具的道路等，我们只需要在新的图像中减去背景就可以得到前景对象了。但是在大多数情况下，我们没有这样的（背景）图像，所以我们需要从我们有的图像中提取背景。如果图像中的交通工具还有影子的话，那这个工作就更难了，因为影子也在移动，仅仅使用背景差分法会把影子也当成前景，这样就有点麻烦了。

为了实现这个目的，科学家们已经提出了几种算法。OpenCV 中已经包含了其中三种比较容易使用的方法。我们将逐一学习到它们。
### BackgroundSubtractorMOG

这是一个以混合高斯模型为基础的前景/背景分割算法。2001年，P. KadewTraKuPong 和 R. Bowden在论文"An improved adaptive background mixture model for real-time tracking with shadow detection" 中进行了介绍。它使用 K（K=3 或 5）个高斯分布混合对背景像素进行建模。混合的权重表示这些颜色停留在场景中的时间比例，背景颜色是那些保持更长时间和更静态的颜色。

在编写代码时，我们需要使用函数：`cv2.createBackgroundSubtractorMOG()`创建一个背景对象。这个函数有些可选参数，比如要进行建模场景的时间长度，高斯混合成分的数量，阈值等。将他们全部设置为默认值。然后在整个视频中我们是需要使用 `backgroundsubtractor.apply() `就可以得到前景的掩码图。

下面是一个简单的例子：

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vtest.avi')
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
        
cap.release()
cv.destroyAllWindows()
```

（所有结果都显示在最后比较）

### BackgroundSubtractorMOG2

这个也是以高斯混合模型为基础的背景/前景分割算法。它是以 2004 年和 2006 年 Z.Zivkovic 的两篇文章为基础的，分别是"Improved adaptive Gaussian mixture model for background subtraction" 和 "Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction" 。这个算法的一个特点是它为每一个像素选择一个合适数目的高斯分布。（上一个方法中我们使用是 K 高斯分布）。它能更好地适应光照不同等各种场景。
和前面一样我们需要创建一个背景对象。但在这里我们我们可以选择是否检测阴影。如果 detectShadows = True（默认值），它就会检测并将影子标记出来，但是这样做会降低处理速度。影子会被标记为灰色。

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vtest.avi')

fgbg = cv.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
        
cap.release()
cv.destroyAllWindows()
```

（最后给出的结果）

### BackgroundSubtractorGMG

此算法结合了静态背景图像估计和每个像素的贝叶斯分割。这是 2012 年Andrew_B.Godbehere，Akihiro_Matsukawa 和 Ken_Goldberg 在文章"Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive Audio Art Installation"中提出的。
它使用前面很少的图像（默认为前 120 帧）进行背景建模。它采用概率前景分割算法，使用贝叶斯推断识别可能的前景对象。这是一种自适应的估计，新观察到的对象比旧的对象具有更高的权重，从而对光照变化产生适应。一些形态学操作如开运算闭运算等被用来除去不需要的噪音。在开始的前几帧图像中你会看到一个黑色窗口。

对结果进行形态学开运算对与去除噪声很有帮助。

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vtest.avi')

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
fgbg = cv.bgsegm.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    cv.imshow('frame',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
        
cap.release()
cv.destroyAllWindows()
```

## 结果
### 原始图片
下图是一个视频的第200帧

![image9](https://docs.opencv.org/4.0.0/resframe.jpg)

### BackgroundSubtractorMOG的结果

![image10](https://docs.opencv.org/4.0.0/resmog.jpg)

### BackgroundSubtractorMOG2的结果

灰色区域显示阴影区域。

![image11](https://docs.opencv.org/4.0.0/resmog2.jpg)

### BackgroundSubtractorGMG的结果

通过形态开口消除噪音。
![image12](https://docs.opencv.org/4.0.0/resgmg.jpg)