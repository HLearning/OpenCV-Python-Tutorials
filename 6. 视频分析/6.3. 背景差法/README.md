## 目标：
本章节你需要学习以下内容:
- 我们将熟悉OpenCV中可用的背景消除方法。

## 基础
在很多基础应用中背景检出都是一个非常重要的步骤。例如顾客统计，使用一个静态摄像头来记录进入和离开房间的人数，或者是交通摄像头，需要提取交通工具的信息等。在所有的这些例子中，首先要将人或车单独提取出来。技术上来说，我们需要从静止的背景中提取移动的前景。

如果你有一张背景（仅有背景不含前景）图像，比如没有顾客的房间，没有交通工具的道路等，那就好办了。我们只需要在新的图像中减去背景就可以得到前景对象了。但是在大多数情况下，我们没有这样的（背景）图像，所以我们需要从我们有的图像中提取背景。如果图像中的交通工具还有影子的话，那这个工作就更难了，因为影子也在移动，仅仅使用减法会把影子也当成前景。真是一件很复杂的事情。

为了实现这个目的科学家们已经提出了几种算法。OpenCV 中已经包含了其中三种比较容易使用的方法。我们将逐一学习到它们。

### BackgroundSubtractorMOG

这是一个以混合高斯模型为基础的前景/背景分割算法。它是 P.KadewTraKuPong和 R.Bowden 在 2001 年提出的。它使用 K（K=3 或 5）个高斯分布混合对背景像素进行建模。使用这些颜色（在整个视频中）存在时间的长短作为混合的权重。背景的颜色一般持续的时间最长，而且更加静止。一个像素怎么会有分布呢？在 x，y 平面上一个像素就是一个像素没有分布，但是我们现在讲的背景建模是基于时间序列的，因此每一个像素点所在的位置在整个时间序列中就会有很多值，从而构成一个分布。

在编写代码时，我们需要使用函数：cv2.createBackgroundSubtractorMOG()创建一个背景对象。这个函数有些可选参数，比如要进行建模场景的时间长度，高斯混合成分的数量，阈值等。将他们全部设置为默认值。然后在整个视频中我们是需要使用 backgroundsubtractor.apply() 就可以得到前景的掩模了。

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

（所有结果都显示在最后以供比较）。

### BackgroundSubtractorMOG2

这个也是以高斯混合模型为基础的背景/前景分割算法。它是以 2004 年和 2006 年 Z.Zivkovic 的两篇文章为基础的。这个算法的一个特点是它为每一个像素选择一个合适数目的高斯分布。（上一个方法中我们使用是 K 高斯分布）。这样就会对由于亮度等发生变化引起的场景变化产生更好的适应。

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

此算法结合了静态背景图像估计和每个像素的贝叶斯分割。这是 2012 年Andrew_B.Godbehere，Akihiro_Matsukawa 和 Ken_Goldberg 在文章中提出的。

它使用前面很少的图像（默认为前 120 帧）进行背景建模。使用了概率前景估计算法（使用贝叶斯估计鉴定前景）。这是一种自适应的估计，新观察到的对象比旧的对象具有更高的权重，从而对光照变化产生适应。一些形态学操作如开运算闭运算等被用来除去不需要的噪音。在前几帧图像中你会得到一个黑色窗口。

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
### 原始框架

下图显示了视频的第200帧

![image9](https://docs.opencv.org/4.0.0/resframe.jpg)

### BackgroundSubtractorMOG的结果

![image10](https://docs.opencv.org/4.0.0/resmog.jpg)

### BackgroundSubtractorMOG2的结果

灰色区域显示阴影区域。

![image11](https://docs.opencv.org/4.0.0/resmog2.jpg)

### BackgroundSubtractorGMG的结果

通过形态开口消除噪音。
![image12](https://docs.opencv.org/4.0.0/resgmg.jpg)