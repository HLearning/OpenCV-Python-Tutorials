## 目标：
- 查找图像渐变，边缘等
- 学习函数：`cv.Sobel()`，`cv.Scharr()`，`cv.Laplacian()`等

## 理论

OpenCV提供三种类型的梯度滤波器或高通滤波器，Sobel，Scharr和Laplacian。我们会一一介绍他们。

Sobel，Scharr 其实就是求一阶或二阶导数。Scharr是对Sobel（使用小的卷积核求解求解梯度角度时）的优化。Laplacian 是求二阶导数。

### Sobel算子和Scharr算子

Sobel算子是高斯联合平滑加微分运算，因此它更能抵抗噪声。你可以指定要采用的导数的方向，垂直或水平（分别通过参数，yorder和xorder），你还可以通过参数ksize指定卷积核的大小。如果ksize = -1，则使用3x3的Scharr滤波器，其结果优于3x3的Sobel滤波器。请参阅所用卷积核的文档。

### Laplacian算子

它计算由关系给出的图像的拉普拉斯算子，$\Delta src= \frac{\partial ^{2}src}{\partial x^{2}}+ \frac{\partial ^{2}src}{\partial y^{2}}$，其中使用Sobel导数找到每个导数。 如果ksize = 1，则使用以下卷积核进行过滤：

$$kernel=\begin{bmatrix}
\ 0\ \ \ \ 1\ \ \ \ 0\\ 
\ 1\ -4\ \ 1\\ 
\ 0\ \ \ \ 1\ \ \ \ 0 
\end{bmatrix}$$

## 代码实现

下面的代码显示了单个图表中的所有运算符，所有卷积核都是5x5大小。输出图像的深度为-1，以获得np.uint8类型的结果。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('dave.jpg',0)

laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
```

窗口将如下图显示：

![image22](https://docs.opencv.org/4.0.0/gradients.jpg)

## 一个重要的事情

在我们的上一个示例中，输出数据类型为cv.CV_8U或np.uint8，但是这有一个小问题，将黑到白转换视为正斜率（它具有正值），而将白到黑转换视为负斜率（它具有负值）。因此，当你将数据转换为np.uint8时，所有负斜率都为零。简单来说，你丢掉了所有的边界。

如果要检测两个边，更好的选择是将输出数据类型保持为某些更高的形式，如cv.CV_16S，cv.CV_64F等，取其绝对值，然后转换回cv.CV_8U。下面的代码演示了水平Sobel滤波器的这个过程以及结果的差异。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('box.png',0)

# Output dtype = cv.CV_8U
sobelx8u = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)

# Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()
```

窗口将如下图显示：
![image23](https://docs.opencv.org/4.0.0/double_edge.jpg)

