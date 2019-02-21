## 目标：
- 学习不同的形态学操作，如侵蚀，膨胀，开放，关闭等
- 学习不同的函数，如：`cv.erode()`，`cv.dilate()`，`cv.morphologyEx()`等

## 理论

形态学转换是基于图像形状的一些简单操作。它通常在二进制图像上执行。它需要两个输入参数，一个是我们的原始图像，第二个是称为结构元素或核，它决定了操作的性质。腐蚀和膨胀是两个基本的形态学运算符。然后它的变体形式如开运算，闭运算，梯度等也发挥作用。我们将在以下图片的帮助下逐一看到它们：

![image14](https://docs.opencv.org/4.0.0/j.png)

## 腐蚀

腐蚀的基本思想就像土壤侵蚀一样，它会腐蚀前景物体的边界（总是试图保持前景为白色）。它是如何做到的呢？卷积核在图像中滑动（如在2D卷积中），只有当卷积核下的所有像素都是1时，原始图像中的像素（1或0）才会被认为是1，否则它会被腐蚀（变为零）。

所以腐蚀作用后，边界附近的所有像素都将被丢弃，具体取决于卷积核的大小。因此，前景对象的厚度或大小减小，或者图像中的白色区域减小。它有助于消除小的白噪声（正如我们在色彩空间章节中看到的那样），或者分离两个连接的对象等。

在这里，作为一个例子，我将使用一个5x5卷积核，其中包含完整的卷积核。让我们看看它是如何工作的：

```python
import cv2 as cv
import numpy as np

img = cv.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)
```
窗口将如下图显示：

![image15](https://docs.opencv.org/4.0.0/erosion.png)

## 膨胀

它恰好与腐蚀相反。这里，如果卷积核下的像素至少一个像素为“1”，则像素元素为“1”。因此它增加了图像中的白色区域或前景对象的大小。通常，在去除噪音的情况下，侵蚀之后是扩张。因为，侵蚀会消除白噪声，但它也会缩小我们的物体,所以我们扩大它。由于噪音消失了，它们不会再回来，但我们的物体区域会增加。它也可用于连接对象的破碎部分。

```python
dilation = cv.dilate(img,kernel,iterations = 1)
```

窗口将如下图显示：

![image16](https://docs.opencv.org/4.0.0/dilation.png)

## 开运算

开运算是先腐蚀后膨胀的合成步骤。如上所述，它有助于消除噪音。这里我们使用函数cv.morphologyEx()

```python
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
```

窗口将如下图显示：

![image17](https://docs.opencv.org/4.0.0/opening.png)

## 闭运算

闭运算与开运算相反，他是先膨胀后腐蚀的操作。它可用于过滤前景对象内的小孔或对象上的小黑点。

```python
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
```

窗口将如下图显示：

![image18](https://docs.opencv.org/4.0.0/closing.png)

## 形态学梯度

它的处理结果是显示膨胀和腐蚀之间的差异。

结果看起来像对象的轮廓。

```python
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
```

窗口将如下图显示：

![image19](https://docs.opencv.org/4.0.0/gradient.png)

## 礼帽

它的处理结果是输入图像和开运算之间的区别。下面的示例是针对9x9卷积核完成的。

```python
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
```

窗口将如下图显示：

![image20](https://docs.opencv.org/4.0.0/tophat.png)

## 黑帽

它是输入图像闭运算和输入图像之间的差异。

```python
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
```

窗口将如下图显示：

![image21](https://docs.opencv.org/4.0.0/blackhat.png)

## 结构元素

我们在Numpy的帮助下手动创建了前面示例中的结构元素。它是正方形的，但在某些情况下可能需要椭圆或圆形卷积核。所以为此，OpenCV有一个函数cv.getStructuringElement()。只需传递卷积核的形状和大小，即可获得所需的卷积核。

```python
# Rectangular Kernel
>>> cv.getStructuringElement(cv.MORPH_RECT,(5,5))
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=uint8)

# Elliptical Kernel
>>> cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0]], dtype=uint8)

# Cross-shaped Kernel
>>> cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
array([[0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]], dtype=uint8)
```