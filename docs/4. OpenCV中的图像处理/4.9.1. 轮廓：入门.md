## 目标：
- 了解什么是轮廓
- 学习查找轮廓，绘制轮廓
- 函数: `cv.findContours()`, `cv.drawContours()`

## 什么是轮廓？
轮廓可以简单地解释为连接所有具有相同的颜色或强度的连续点（沿着边界）的曲线。轮廓是形状分析和物体检测和识别的很有用的工具。

- 为了更好的准确性，使用二进制图像，因此，在找到轮廓之前，应用阈值或canny边缘检测。
- 从OpenCV 3.2开始，findContours()不再修改源图像，而是将修改后的图像作为三个返回参数中的第一个返回。
- 在OpenCV中，找到轮廓就像从黑色背景中找到白色物体。所以请记住，要找到的对象应该是白色，背景应该是黑色。

让我们看看如何找到二进制图像的轮廓：

```python
import numpy as np
import cv2 as cv

im = cv.imread('test.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
```

在`cv.findContours()`函数中有三个参数，第一个是源图像，第二个是轮廓检索模式，第三个是轮廓逼近方法。它输出轮廓和层次结构。contours是图像中所有轮廓的Python列表，每个单独的轮廓是对象边界点坐标(x,y)的Numpy数组。

> 注意：我们稍后将详细讨论第二和第三个参数以及层次结构。在此之前，代码示例中给出的值对所有图像都可以正常工作。

## 如何绘制轮廓？

要绘制轮廓，可以使用`cv.drawContours`函数。如果图像有边界点，它也可以用于绘制任何形状。它的第一个参数是源图像，第二个参数是应该作为Python列表传递的轮廓，第三个参数是轮廓索引（在绘制单个轮廓时很有用。绘制所有轮廓，传递-1），其余参数是颜色，厚度等等

要绘制图像中的所有轮廓：

```python
cv.drawContours(img, contours, -1, (0,255,0), 3)
```

要绘制单个轮廓，请输入四个轮廓点：

```python
cv.drawContours(img, contours, 3, (0,255,0), 3)
```

但大多数时候，下面的方法会很有用：

```python
cnt = contours[4]
cv.drawContours(img, [cnt], 0, (0,255,0), 3)
```

> 注意：最后两种方法是相同的，但是当你继续前进时，你会发现最后一种方法更有用。

## 轮廓近似方法

这是`cv.findContours`函数中的第三个参数。它实际上表示什么？

在上面，我们告诉轮廓是具有相同强度的形状的边界。它存储形状边界的（x，y）坐标。但是它存储了所有坐标吗？这由该轮廓近似方法指定。

如果传递`cv.CHAIN_APPROX_NONE`，则存储所有边界点。但实际上我们需要所有的积分吗？例如，你找到了直线的轮廓，你是否需要线上的所有点来表示该线？不，我们只需要该线的两个端点。这就是`cv.CHAIN_APPROX_SIMPLE`的作用。它删除所有冗余点并压缩轮廓，从而节省内存。

下面的矩形图像展示了这种技术。只需在轮廓阵列中的所有坐标上绘制一个圆圈（以蓝色绘制）。第一张图片显示了我用`cv.CHAIN_APPROX_NONE`（734点）获得的点数，第二张图片显示了一张带有`cv.CHAIN_APPROX_SIMPLE`（仅4点）的点数，它节省了不少内存！

![image31](https://docs.opencv.org/4.0.0/none.jpg)