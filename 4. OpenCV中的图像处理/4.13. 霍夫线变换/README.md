## 目标：
本章节你需要学习以下内容:
- 我们将理解霍夫变换的概念。
- 我们将看到如何使用它来检测图像中的线条。
- 我们将看到以下函数：`cv.HoughLines()`，`cv.HoughLinesP()`

## 理论
如果你能够以数学形式表示该形状，则霍夫变换是一种检测任何形状的流行技术。它可以检测形状，即使它被破坏或扭曲一点点。我们将看到它如何适用于生产线。

线可以表示为$y=mx+c$或以参数形式表示为$\rho =x\ cos\theta +y\ sin\theta$其中$\rho$是从原点到线的垂直距离，$\theta$是由该垂直线和水平轴形成的角度 以逆时针方向测量（该方向因你表示坐标系的方式而异。此表示在OpenCV中使用）。检查下图：

![image69](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/4.Image%20Processing%20in%20OpenCV/Image/image69.png)

因此，如果线在原点以下通过，它将具有正rho和小于180的角度。如果它超过原点，而不是采用大于180的角度，则角度小于180，并且rho被认为是否定的。任何垂直线都有0度，水平线有90度。

现在让我们看看霍夫变换如何为线条工作。任何线都可以用这两个术语表示，$\left ( \rho ,\theta  \right )$。因此，首先它创建一个2D数组或累加器（以保存两个参数的值），并且最初设置为0。令行表示$\rho$，列表示$\theta$。阵列的大小取决于你需要的准确度。假设你希望角度精度为1度，则需要180列。对于$\rho$，可能的最大距离是图像的对角线长度。因此，取一个像素精度，行数可以是图像的对角线长度。

考虑一个100x100的图像，中间有一条水平线。取第一点。你知道它的（x，y）值。现在在线方程中，将值$\theta= 0,1,2,\cdots ,180$并检查你得到的$\rho$。对于每个$\left ( \rho ,\theta  \right )$对，在我们的累加器中将其在相应的$\left ( \rho ,\theta  \right )$单元格中增加1。所以现在在累加器中，单元格（50,90）= 1以及其他一些单元格。

现在取第二点就行了。和上面一样。增加与你获得的（rho，theta）对应的单元格中的值。这次，单元格（50,90）= 2.你实际做的是投票给$\left ( \rho ,\theta  \right )$值。你可以继续执行此过程中的每个点。在每个点，单元格（50,90）将递增或投票，而其他单元格可能会或可能不会被投票。这样，最后，单元格（50,90）将获得最大票数。因此，如果你在累加器中搜索最大投票数，则会得到值（50,90），表示此图像中距离原点和角度为90度的距离为50。它在下面的动画中有很好的展示（图片提供：Amos Storkey）

![image70](https://docs.opencv.org/4.0.0/houghlinesdemo.gif)

这就是霍夫变换对线条的作用。 它很简单，也许你可以自己使用Numpy来实现它。 下面是显示累加器的图像。 某些位置的亮点表示它们是图像中可能线条的参数。 （图片提供：维基百科）

![image71](https://docs.opencv.org/4.0.0/houghlines2.jpg)

## OpenCV中的霍夫变换

上面解释的所有内容都封装在OpenCV函数cv.HoughLines（）中。 它只返回一个数组：math：（rho，theta）`values。$\rho$以像素为单位测量，$\theta$以弧度为单位测量。第一个参数，输入图像应该是二进制图像，因此在应用霍夫变换之前应用阈值或使用精确边缘检测。 第二和第三参数分别是$\rho$和$\theta$精度。第四个参数是阈值，这意味着它应该被视为一条线的最小投票。请记住，投票数取决于该线上的点数。因此它表示应检测的最小行长度。

```python
import cv2 as cv
import numpy as np
img = cv.imread('../data/sudoku.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)
lines = cv.HoughLines(edges,1,np.pi/180,200)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
cv.imwrite('houghlines3.jpg',img)
```

窗口将如下图显示：

![image72](https://docs.opencv.org/4.0.0/houghlines3.jpg)

## 概率Hough变换

在霍夫变换中，你可以看到即使对于具有两个参数的行，也需要大量计算。概率Hough变换是我们看到的Hough变换的优化。它没有考虑所有要点。相反，它只需要一个足以进行线检测的随机点子集。我们必须降低门槛。 请参见下图，其中比较霍夫空间中的霍夫变换和概率霍夫变换。（图片提供：Franck Bettinger的主页）

![image73](https://docs.opencv.org/4.0.0/houghlines4.png)

OpenCV实现基于使用Matas，J。和Galambos，C。和Kittler，J.V。[122]的渐进概率Hough变换的线的鲁棒检测。 使用的函数是cv.HoughLinesP（）。 它有两个新的论点。

* minLineLength - 最小线长。 短于此的线段将被拒绝。
* maxLineGap - 线段之间允许的最大间隙，将它们视为一条线。

最好的是，它直接返回行的两个端点。在前面的例子中，你只得到了行的参数，你必须找到所有的点。在这里，一切都是直接而简单的。

```python
import cv2 as cv
import numpy as np
img = cv.imread('../data/sudoku.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)
lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
cv.imwrite('houghlines5.jpg',img)
```

窗口将如下图显示：

![image74](https://docs.opencv.org/4.0.0/houghlines5.jpg)

