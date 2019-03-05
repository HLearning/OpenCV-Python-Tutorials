### 3、2D直方图

#### （1）介绍

在第一篇文章中，我们计算并绘制了一维直方图。 它被称为一维，因为我们只考虑一个特征，即像素的灰度强度值。 但在二维直方图中，你考虑两个特征。 通常，它用于查找颜色直方图，其中两个要素是每个像素的色调和饱和度值。

有一个python样本（samples / python / color_histogram.py）已经用于查找颜色直方图。 我们将尝试了解如何创建这样的颜色直方图，它将有助于理解直方图反投影等其他主题。

#### （2）OpenCV中的2D直方图

它很简单，使用相同的函数cv.calcHist（）计算。 对于颜色直方图，我们需要将图像从BGR转换为HSV。 （请记住，对于1D直方图，我们从BGR转换为灰度）。 对于2D直方图，其参数将修改如下：

* channels = [0,1]因为我们需要处理H和S平面。
* b = H平面为[180,256] 180，S平面为256。
* range = [0,180,0,256] Hue值介于0和180之间，饱和度介于0和256之间。

现在检查以下代码：

```python
import numpy as np
import cv2 as cv
img = cv.imread('home.jpg')
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
```

#### （3）Numpy中的2D直方图

Numpy还为此提供了一个特定的功能：np.histogram2d（）。 （请记住，对于1D直方图，我们使用np.histogram（））。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('home.jpg')
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
hist, xbins, ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])
```

第一个参数是H平面，第二个是S平面，第三个是每个箱子的数量，第四个是它们的范围。

现在我们可以检查如何绘制这种颜色直方图。

#### （4）绘制2D直方图

方法 - 1：使用cv.imshow（）

我们得到的结果是一个大小为180x256的二维数组。 因此我们可以像使用cv.imshow（）函数一样正常显示它们。 它将是一个灰度图像，除非你知道不同颜色的色调值，否则它不会过多地了解那里的颜色。

方法-2：使用Matplotlib

我们可以使用matplotlib.pyplot.imshow（）函数绘制具有不同颜色图的2D直方图。 它让我们更好地了解不同的像素密度。 但是，除非你知道不同颜色的色调值，否则这也不会让我们知道第一眼看到的是什么颜色。 我还是喜欢这种方法。 它简单而且更好。

**注意：在使用此功能时，请记住，插值标志应该最接近以获得更好的结果。**

考虑代码：

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('home.jpg')
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
hist = cv.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
plt.imshow(hist,interpolation = 'nearest')
plt.show()
```

下面是输入图像及其颜色直方图。 X轴显示S值，Y轴显示Hue。

![image54](https://docs.opencv.org/4.0.0/2dhist_matplotlib.jpg)

在直方图中，你可以看到H = 100和S = 200附近的一些高值。它对应于天空的蓝色。 类似地，在H = 25和S = 100附近可以看到另一个峰值。它对应于宫殿的黄色。 你可以使用任何图像编辑工具（如GIMP）对其进行验证。

方法3：OpenCV样本风格!!

在OpenCV-Python2样本中有一个颜色直方图的示例代码（samples / python / color_histogram.py）。 如果运行代码，则可以看到直方图也显示相应的颜色。 或者只是输出颜色编码的直方图。 它的结果非常好（虽然你需要添加额外的一堆线）。

在该代码中，作者在HSV中创建了一个颜色映射。 然后将其转换为BGR。 得到的直方图图像与该颜色图相乘。 他还使用一些预处理步骤来移除小的孤立像素，从而产生良好的直方图。

我把它留给读者来运行代码，分析它并拥有自己的hack arounds。 以下是与上述相同图像的代码输出：

![image55](https://docs.opencv.org/4.0.0/2dhist_opencv.jpg)

你可以在直方图中清楚地看到存在哪些颜色，蓝色存在，黄色存在，并且由于棋盘存在一些白色。 很好!!!