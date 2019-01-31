 ## 目标
本章节你需要学习以下内容:
- 将不同的几何变换应用于图像，如平移，旋转，仿射变换等。
- 你将看到以下函数：`cv.getPerspectiveTransform`

## 转换
OpenCV提供了两个转换函数`cv.warpAffine`和`cv.warpPerspective`，你可以使用它们进行各种转换。`cv.warpAffine`采用2x3变换矩阵作为参数输入，而`cv.warpPerspective`采用3x3变换矩阵作为参数输入。

## 缩放
缩放只是调整图像大小。为此，OpenCV附带了一个函数`cv.resize()`。可以手动指定图像的大小，也可以指定缩放系数。可以使用不同的插值方法，常用的插值方法是用于缩小的`cv.INTER_AREA`和用于缩放的`cv.INTER_CUBIC`（慢）和`cv.INTER_LINEAR`。默认情况下，使用的插值方法是`cv.INTER_LINEAR`，它用于所有调整大小的操作。你可以使用以下方法之一调整输入图像的大小：

```python
import numpy as np
import cv2 as cv

img = cv.imread('messi5.jpg')

res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)

#OR

height, width = img.shape[:2]
res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
```
## 平移
平移是对象位置的移动。如果你知道像素点(x，y)要位移的距离，让它为变为(`$ t_x $`,`$ t_y $`)，你可以创建变换矩阵**M**，如下所示：

`$ M=
\begin{bmatrix}
1&0&t_x\\
0&1&t_y
\end{bmatrix} $`

你可以将其设置为np.float32类型的Numpy数组，并将其传递给cv.warpAffine()函数。下面的示例演示图像像素点整体进行(100,50)位移：

```python
import numpy as np
import cv2 as cv

img = cv.imread('messi5.jpg',0)
rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()
```

> 注意：cv.warpAffine()函数的第三个参数是输出图像的大小，它应该是(宽度，高度)的形式。请记住,width=列数，height=行数。

窗口将如下图显示：
![image2](https://docs.opencv.org/4.0.0/translation.jpg)

## 旋转

通过改变图像矩阵实现图像旋转角度θ

`$ M=\begin{bmatrix}
cos\Theta &-sin\Theta\\ 
sin\Theta & cos\Theta 
\end{bmatrix} $`

但OpenCV提供可调旋转，即旋转中心可调，因此你可以在任何位置进行旋转。修正的变换矩阵由下式给出：

`$ \begin{bmatrix}
\alpha  & \beta & \left ( 1-\alpha  \right )\cdot center.x-\beta \cdot center.y \\ 
-\beta   & \alpha & \beta \cdot center.x\left ( 1-\alpha  \right )\cdot center.y
\end{bmatrix} $`

其中：

`$ \alpha = scale\cdot cos\Theta $`

`$ \beta = scale\cdot sin\Theta $`

为了找到这个转换矩阵，OpenCV提供了一个函数cv.getRotationMatrix2D。以下示例将图像相对于中心旋转90度而不进行任何缩放。

```python
img = cv.imread('messi5.jpg',0)
rows,cols = img.shape

M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv.warpAffine(img,M,(cols,rows))
```

窗口将如下图显示：

![image3](https://docs.opencv.org/4.0.0/rotation.jpg)

## 仿射变换

在仿射变换中，原始图像中的所有平行线仍将在输出图像中平行。为了找到变换矩阵，我们需要输入图像中的三个点及其在输出图像中的相应位置。然后cv.getAffineTransform将创建一个2x3矩阵，最后该矩阵将传递给cv.warpAffine。

参考以下示例，并查看我选择的点（以绿色标记）：

```python
img = cv.imread('drawing.png')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv.getAffineTransform(pts1,pts2)

dst = cv.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```

窗口将如下图显示：

![image4](https://docs.opencv.org/4.0.0/affine.jpg)

## 透视变换

对于透视变换，你需要一个3x3变换矩阵。即使在转换之后，直线仍将保持笔直。要找到此变换矩阵，输入图像上需要4个点，输出图像上需要相应的4个点。在这4个点中，其中任意3个不共线。然后可以通过函数cv.getPerspectiveTransform找到变换矩阵，将cv.warpPerspective应用于此3x3变换矩阵。

请参阅以下代码：

```python
img = cv.imread('sudoku.png')
rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv.getPerspectiveTransform(pts1,pts2)

dst = cv.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
```

窗口将如下图显示：

![image5](https://docs.opencv.org/4.0.0/perspective.jpg)