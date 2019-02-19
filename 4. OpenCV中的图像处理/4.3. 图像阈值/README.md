## 目标：
- 学习简单的阈值处理，自适应阈值处理，Otsu's的阈值处理等
- 学习函数：`cv.threshold`，`cv.adaptiveThreshold`等

## 简单阈值处理

这种阈值处理的方法是简单易懂的。如果像素值大于阈值，则为其分配一个值（可以是白色），否则为其分配另一个值（可以是黑色）。使用的函数是cv.threshold。函数第一个参数是源图像，它应该是灰度图像。第二个参数是用于对像素值进行分类的阈值。第三个参数是maxVal，它表示如果像素值大于（有时小于）阈值则要给出的值。OpenCV提供不同类型的阈值，由函数的第四个参数决定。不同的类型有：

- cv.THRESH_BINARY
- cv.THRESH_BINARY_INV
- cv.THRESH_TRUNC
- cv.THRESH_TOZERO
- cv.THRESH_TOZERO_INV

文档清楚地解释了每种类型的含义。

函数将获得两个输出。第一个是retavl，将在后面解释它的作用。第二个输出是我们的阈值图像。

参考以下代码：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('gradient.png',0)
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    
plt.show()
```

> 注意：为了绘制多个图像，我们使用了plt.subplot()函数。请查看Matplotlib文档以获取更多详细信息

窗口将如下图显示：

![image6](https://docs.opencv.org/4.0.0/threshold.jpg)

## 自适应阈值处理

在上面，我们使用全局值作为阈值，但在图像在不同区域具有不同照明条件的所有条件下可能并不好。在那种情况下，我们进行自适应阈值处理，算法计算图像的小区域的阈值，所以我们对同一幅图像的不同区域给出不同的阈值，这给我们在不同光照下的图像提供了更好的结果。

这种阈值处理方法有三个指定输入参数和一个输出参数。

**Adaptive Method** - 自适应方法，决定如何计算阈值。

* cv.ADAPTIVE_THRESH_MEAN_C：阈值是邻域的平均值。
* cv.ADAPTIVE_THRESH_GAUSSIAN_C：阈值是邻域值的加权和，其中权重是高斯窗口。

**Block Size** - 邻域大小，它决定了阈值区域的大小。

**C** - 它只是从计算的平均值或加权平均值中减去的常数。

下面的代码比较了具有不同照明的图像的全局阈值处理和自适应阈值处理：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('sudoku.png',0)
img = cv.medianBlur(img,5)

ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    
plt.show()
```

窗口将如下图显示：

![image7](https://docs.opencv.org/4.0.0/ada_threshold.jpg)

## Otsu's 二值化

在第一节中，我只告诉你另一个参数是retVal，但没告诉你它的作用。其实，它是用来进行Otsu's二值化。

在全局阈值处理中，我们使用任意值作为阈值，那么，我们如何知道我们选择的值是好还是不好？答案是，试错法。但如果是双峰图像（简单来说，双峰图像是直方图有两个峰值的图像）我们可以将这些峰值中间的值近似作为阈值，这就是Otsu二值化的作用。简单来说，它会根据双峰图像的图像直方图自动计算阈值。（对于非双峰图像，二值化不准确。）

为此，使用了我们的cv.threshold()函数，但是需要多传递一个参数cv.THRESH_OTSU。这时要吧阈值设为零。然后算法找到最佳阈值并返回第二个输出retVal。如果未使用Otsu二值化，则retVal与你设定的阈值相同。

请查看以下示例。输入图像是嘈杂的图像。在第一种情况下，我将全局阈值应用为值127。在第二种情况下，我直接应用了Otsu的二值化。在第三种情况下，我使用5x5高斯卷积核过滤图像以消除噪声，然后应用Otsu阈值处理。来看看噪声过滤如何改善结果。

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('noisy2.png',0)

# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    
plt.show()
```

窗口将如下图显示：

![image8](https://docs.opencv.org/4.0.0/otsu.jpg)

下面讲解了Otsu二值化的Python实现，以展示它的实际工作原理。如果你不感兴趣，可以跳过这个内容。

由于我们正在使用双峰图像，因此Otsu的算法试图找到一个阈值(t)，它最小化了由关系给出的加权类内方差：

$$ \sigma _{w}^{2}\left ( t \right )= q_{1}\left ( t \right )\sigma _{1}^{2}\left ( t \right )+q_{2}\left ( t \right )\sigma _{2}^{2}\left ( t \right ) $$

其中：

$$ q_{1\left ( t \right )}=\sum_{i=1}^{t}P\left ( i \right ) \ \ \ \ \&\ \ \ \ q_{1\left ( t \right )}=\sum_{i=1}^{I}P\left ( i \right ) $$

$$ \mu _{1}\left ( t \right )=\sum_{t}^{i=1}\frac{iP\left ( i \right )}{q_{1}\left ( t \right )}\ \ \ \ \ \ \&\ \ \ \ \ \ \mu _{2}\left ( t \right )=\sum_{I}^{i=t+1}\frac{iP\left ( i \right )}{q_{2}\left ( t \right )} $$

$$ \sigma _{1}^{2}\left ( t \right )=\sum_{i=1}^{t}\left [ i - \mu _{1} \left ( t \right )\right ]^{2}\frac{P\left ( i \right )}{q_{1}\left ( t \right )}\ \ \ \ \&\ \ \ \ \sigma _{2}^{2}\left ( t \right )=\sum_{i=t+1}^{I}\left [ i - \mu _{1} \left ( t \right )\right ]^{2}\frac{P\left ( i \right )}{q_{2}\left ( t \right )} $$

它实际上找到了一个位于两个峰之间的t值，这样两个类的方差都是最小的。它可以简单地在Python中实现，如下所示：

```python
img = cv.imread('noisy2.png',0)
blur = cv.GaussianBlur(img,(5,5),0)

# find normalized_histogram, and its cumulative distribution function
hist = cv.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()

bins = np.arange(256)

fn_min = np.inf
thresh = -1

for i in xrange(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    b1,b2 = np.hsplit(bins,[i]) # weights
    
    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
    
    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i
        
# find otsu's threshold value with OpenCV function
ret, otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
print( "{} {}".format(thresh,ret) )
```

> 注意：这里的一些功能可能是之前没有讲过的，但我们将在后面的章节中介绍它们
