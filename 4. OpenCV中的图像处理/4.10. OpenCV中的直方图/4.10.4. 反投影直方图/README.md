#### （1）理论

它由Michael J. Swain，Dana H. Ballard在他们的论文“通过颜色直方图索引”中提出。
用简单的词语实际上是什么？它用于图像分割或查找图像中感兴趣的对象。简单来说，它会创建一个与输入图像大小相同（但是单个通道）的图像，其中每个像素对应于该像素属于我们对象的概率。在更简单的世界中，与剩余部分相比，输出图像将使我们感兴趣的对象更白。嗯，这是一个直观的解释。 （我不能让它变得更简单）。直方图反投影与camshift算法等一起使用。

我们该怎么做呢 ？我们创建一个包含我们感兴趣对象的图像的直方图（在我们的例子中，是地面，离开玩家和其他东西）。对象应尽可能填充图像以获得更好的结果。并且颜色直方图优于灰度直方图，因为对象的颜色是比其灰度强度更好的定义对象的方式。然后我们将这个直方图“反投影”到我们需要找到对象的测试图像上，换句话说，我们计算每个像素属于地面并显示它的概率。通过适当的阈值处理得到的输出为我们提供了基础。

#### （2）Numpy中的算法

1. 首先，我们需要计算我们需要找到的对象（让它为'M'）和我们要搜索的图像（让它为'我'）的颜色直方图。

```python
import numpy as np
import cv2 as cvfrom matplotlib import pyplot as plt
#roi is the object or region of object we need to find
roi = cv.imread('rose_red.png')
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
#target is the image we search in
target = cv.imread('rose.png')
hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
# Find the histograms using calcHist. Can be done with np.histogram2d also
M = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
I = cv.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
```

2. 求出比率$R=\frac{M}{I}$。然后反投影R，即使用R作为调色板并创建一个新图像，每个像素作为其对应的目标概率。 即B（x，y）= R[h（x，y），s（x，y）]其中h是色调，s是（x，y）处像素的饱和度。 之后应用条件B（x，y）= min [B（x，y），1]。

```python
h,s,v = cv.split(hsvt)
B = R[h.ravel(),s.ravel()]
B = np.minimum(B,1)
B = B.reshape(hsvt.shape[:2])

```

3. 现在应用圆盘的卷积，B = D * B，其中D是盘卷积核。

```python
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(B,-1,disc,B)
B = np.uint8(B)
cv.normalize(B,B,0,255,cv.NORM_MINMAX)
```

4. 现在最大强度的位置为我们提供了物体的位置。 如果我们期望图像中有一个区域，那么对适当值进行阈值处理会得到很好的结果。

```python
ret,thresh = cv.threshold(B,50,255,0)
```

#### （3）OpenCV中的反投影

OpenCV提供了一个内置函数cv.calcBackProject（）。 它的参数与cv.calcHist（）函数几乎相同。 它的一个参数是直方图，它是对象的直方图，我们必须找到它。 此外，在传递给backproject函数之前，应该对象直方图进行规范化。 它返回概率图像。 然后我们将图像与光盘卷积核卷积并应用阈值。 以下是我的代码和输出：

```python
import numpy as np
import cv2 as cv
roi = cv.imread('rose_red.png')
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
target = cv.imread('rose.png')
hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
# calculating object histogram
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
# Now convolute with circular disc
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)
# threshold and binary AND
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))
res = cv.bitwise_and(target,thresh)
res = np.vstack((target,thresh,res))
cv.imwrite('res.jpg',res)
```

以下是我合作过的一个例子。 我使用蓝色矩形内的区域作为样本对象，我想提取完整的地面。

![image56](https://docs.opencv.org/4.0.0/backproject_opencv.jpg)