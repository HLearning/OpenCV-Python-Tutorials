## 理论
直方图是什么？你可以将直方图视为图形或绘图，它可以让你全面了解图像的强度分布。它是在X轴上具有像素值（范围从0到255，并非总是）的图和在Y轴上的图像中的对应像素数。

这只是理解图像的另一种方式。通过查看图像的直方图，你可以直观了解该图像的对比度，亮度，强度分布等。今天几乎所有的图像处理工具都提供了直方图的功能。以下是来自Cambridge in Color网站的图片，我建议你访问该网站了解更多详情。

![image44](https://docs.opencv.org/4.0.0/histogram_sample.jpg)
你可以看到图像及其直方图。（请记住，此直方图是为灰度图像绘制的，而不是彩色图像）。直方图的左区域显示图像中较暗像素的数量，右区域显示较亮像素的数量。从直方图中，你可以看到暗区域不仅仅是更亮的区域，中间色调的数量（中间区域的像素值，比如大约127）非常少。

## 查找直方图
现在我们知道什么是直方图，我们可以研究如何找到它。 OpenCV和Numpy都具有内置功能。在使用这些功能之前，我们需要了解与直方图相关的一些术语。

**BINS**：上面的直方图显示了每个像素值的像素数，即从0到255.即你需要256个值来显示上面的直方图。但是考虑一下，如果你不需要分别找到所有像素值的像素数，但像素值区间的像素数是多少呢？例如，你需要找到介于0到15之间，然后是16到31，......，240到255之间的像素数。你只需要16个值来表示直方图。这就是OpenCV教程中直方图中给出的示例。

所以你要做的只是将整个直方图分成16个子部分，每个子部分的值是其中所有像素数的总和。每个子部分称为“BIN”。在第一种情况下，bin的数量是256（每个像素一个），而在第二种情况下，它只有16. BINS由OpenCV docs中的术语histSize表示。

**DIMS**：这是我们收集数据的参数数量。在这种情况下，我们只收集强度值有关的数据，所以这里是1。

**RANGE**：这是要测量的强度值范围。通常，它是[0,256]，即所有强度值。

### 1. OpenCV中的直方图计算

使用`cv.calcHist()`函数来查找直方图。让我们熟悉一下这个函数及其参数：

`cv.calcHist（images，channels，mask，histSize，ranges [，hist [，accumulate]]）`

1. images：它是uint8或float32类型的源图像。它应该用方括号表示，即“[img]”。
2. channels：它也在方括号中给出。它是我们计算直方图的通道索引。例如，如果输入是灰度图像，则其值为[0]。对于彩色图像，你可以通过[0]，[1]或[2]分别计算蓝色，绿色或红色通道的直方图。
3. mask：掩模图像。要查找完整图像的直方图，它将显示为“无”。但是，如果要查找图像特定区域的直方图，则必须为其创建蒙版图像并将其作为蒙版。 （稍后我会举一个例子。）
4. histSize：这代表我们的BIN计数。需要在方括号中给出。对于满量程，我们通过[256]。
5. ranges：范围。通常是[0,256]。

那么让我们从一个示例图像开始吧。只需以灰度模式加载图像并找到其完整的直方图。

```python
img = cv.imread('home.jpg',0)
hist = cv.calcHist([img],[0],None,[256],[0,256])
```
hist是256x1数组，每个值对应于该图像中具有相应像素值的像素数。

### 2. Numpy中的直方图计算

Numpy还为你提供了一个函数，`np.histogram()`。 因此，你可以尝试以下行代替`calcHist()`函数：

```python
hist,bins = np.histogram(img.ravel(),256,[0,256])
```
hist与我们之前计算的相同。但是bins中有257个元素，因为Numpy中bins计算为0-0.99,1-1.99,2-2.99等。所以最终范围是255-255.99。为了表示这一点，他们还在bins中追加了256。但我们不需要256.高达255就够了。

参考： Numpy有另一个函数，`np.bincount()`，它比`np.histogram()`快得多（大约10倍）。因此，对于一维直方图，你可以更好地尝试。不要忘记在np.bincount中设置minlength = 256。例如，`hist = np.bincount（img.ravel（），minlength = 256）`

> 注意
OpenCV函数比`np.histogram()`快（约40倍）。所以推荐使用OpenCV功能。

## 绘制直方图
有两种方法：
* 简短方法：使用Matplotlib绘图功能
* 复杂方法：使用OpenCV绘图功能

### 1. 使用Matplotlib
Matplotlib附带直方图绘图功能：`matplotlib.pyplot.hist()`
它直接找到直方图并绘制它。你无需使用`calcHist()`或`np.histogram()`函数来查找直方图。请参阅以下代码：

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('home.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()
```

窗口将如下图显示：

![image45](https://docs.opencv.org/4.0.0/histogram_matplotlib.jpg)

或者你可以使用matplotlib的通用画法，这对BGR图有好处。 为此，你需要首先找到直方图数据。 

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('home.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
```

窗口将如下图显示：

![image46](https://docs.opencv.org/4.0.0/histogram_rgb_plot.jpg)

你可以从上图中推断，蓝色在图像中有一些高值区域（显然是因为天空）

### 2. 使用OpenCV
在这里你可以调整直方图的值及其bin值，使其看起来像x，y坐标，这样你就可以使用`cv.line()`或`cv.polyline()`函数绘制它，以生成与上面相同的图像。

## 蒙板的应用
我们使用`cv.calcHist()`来查找完整图像的直方图。 如果要查找图像某些区域的直方图，该怎么办？ 只需在要查找直方图的区域上创建一个白色的蒙版图像，其余为黑色，然后将其作为蒙板传递。

```python
img = cv.imread('home.jpg',0)
# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv.bitwise_and(img,img,mask = mask)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()
```

看到结果。 在直方图中，蓝线显示完整图像的直方图，而绿线显示屏蔽区域的直方图。

![image47](https://docs.opencv.org/4.0.0/histogram_masking.jpg)