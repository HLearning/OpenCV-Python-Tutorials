## 目标：
- 使用各种低通滤波器模糊图像
- 将自定义滤波器应用于图像（2D卷积）

## 2D卷积（图像过滤）
与一维信号一样，图像也可以使用各种低通滤波器（LPF），高通滤波器（HPF）等进行滤波。LPF有助于消除噪声，模糊图像等。HPF滤波器有助于找到图片的边缘。

OpenCV提供了一个函数cv.filter2D()来将卷积核与图像进行卷积。例如，我们将尝试对图像进行平均滤波。下面是一个5x5平均滤波器的核：

$$ K=\frac{1}{25}\begin{bmatrix}
\ 1\ \ 1 \ \ 1 \ \ 1\ \ 1\\ 
\ 1\ \ 1 \ \ 1 \ \ 1\ \ 1\\ 
\ 1\ \ 1 \ \ 1 \ \ 1\ \ 1\\ 
\ 1\ \ 1 \ \ 1 \ \ 1\ \ 1\\ 
\ 1\ \ 1 \ \ 1 \ \ 1\ \ 1 
\end{bmatrix} $$

操作步骤如下：将此核放在一个像素A上，求与核对应的图像上 25（5x5）个像素的和，取其平均值并用新的平均值替换像素A的值。重复以上操作直到将图像的每一个像素值都更新一遍。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('opencv_logo.png')

kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image9](https://docs.opencv.org/4.0.0/filter.jpg)

## 图像模糊（图像平滑）

通过将图像与低通滤波器卷积核卷积来实现平滑图像。它有助于消除噪音，从图像中去除了高频内容（例如：噪声，边缘）。因此在此操作中边缘会模糊一点。（有的平滑技术也不会平滑边缘）。OpenCV主要提供四种平滑技术。

### 1. 均值滤波

这是由一个归一化卷积框完成的。它取卷积核区域下所有像素的平均值并替换中心元素。这是由函数`cv.blur()`或`cv.boxFilter()`完成的。查看文档以获取有关卷积核的更多详细信息。我们应该指定卷积核的宽度和高度，3x3标准化的盒式过滤器如下所示：

$$ K=\frac{1}{9}\begin{bmatrix}
\ 1 \ \ 1\ \ 1\\ 
\ 1 \ \ 1\ \ 1\\ 
\ 1 \ \ 1\ \ 1 
\end{bmatrix} $$

> 注意：如果不想使用规范化的框过滤器，请使用`cv.boxFilter()`。将参数`normalize = False`传递给函数。

使用5x5大小的卷积核检查下面的示例演示：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('opencv-logo-white.png')

blur = cv.blur(img,(5,5))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image10](https://docs.opencv.org/4.0.0/blur.jpg)

### 2. 高斯滤波

下面把卷积核换成高斯核。它是通过函数`cv.GaussianBlur()`完成的。我们应该指定卷积核的宽度和高度，它应该是正数并且是奇数。我们还应该分别指定X和Y方向的标准偏差sigmaX和sigmaY。如果仅指定了sigmaX，则sigmaY与sigmaX相同。如果两者都为零，则根据卷积核大小计算它们。高斯模糊在从图像中去除高斯噪声方面非常有效。

如果需要，可以使用函数cv.getGaussianKernel()创建高斯卷积核。

上面的代码可以修改为高斯模糊：

```python
blur = cv.GaussianBlur(img,(5,5),0)
```

窗口将如下图显示：

![image11](https://docs.opencv.org/4.0.0/gaussian.jpg)

### 3. 中值滤波
顾名思义，函数`cv.medianBlur()`取卷积核区域下所有像素的中值，并用该中值替换中心元素。这对去除图像中的椒盐噪声非常有效。有趣的是，在上述滤波器中，中心元素是新计算的值，其可以是图像中的像素值或新值。但在中值模糊中，中心元素总是被图像中的某个像素值替换,它有效地降低了噪音。其卷积核大小应为正整数。

在这个演示中，我为原始图像添加了50％的噪点并应用了中值模糊。检查结果：

```python
median = cv.medianBlur(img,5)
```

窗口将如下图显示：

![image12](https://docs.opencv.org/4.0.0/median.jpg)

### 4. 双边过滤
`cv.bilateralFilter()`在降低噪音方面非常有效，同时保持边缘清晰。但与其他过滤器相比，操作速度较慢。我们已经看到高斯滤波器采用像素周围的邻域并找到其高斯加权平均值。该高斯滤波器仅是空间的函数，即在滤波时考虑附近的像素。它没有考虑像素是否具有几乎相同的强度。它不考虑像素是否是边缘像素。所以它也模糊了边缘，我们不想这样做。

双边滤波器在空间中也采用高斯滤波器，但是还有一个高斯滤波器是像素差的函数。空间的高斯函数确保仅考虑附近的像素用于模糊，而强度差的高斯函数确保仅考虑具有与中心像素相似的强度的像素用于模糊。因此它保留了边缘，因为边缘处的像素将具有较大的强度变化。

下面的示例显示使用双边过滤器

```python
blur = cv.bilateralFilter(img,9,75,75)
```

窗口将如下图显示：

![image13](https://docs.opencv.org/4.0.0/bilateral.jpg)
