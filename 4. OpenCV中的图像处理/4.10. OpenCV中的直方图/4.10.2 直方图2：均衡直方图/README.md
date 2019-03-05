## 目标
- 学习直方图均衡化的概念，并利用它来提高图像对比度

## 理论
考虑一个像素值仅限于某些特定值范围的图像。 例如，较亮的图像将所有像素限制为高值。 但是，良好的图像将具有来自图像的所有区域的像素。 所以你需要将这个直方图拉伸到两端（如下图所示，来自维基百科），这就是直方图均衡所做的，通常可以改善图像的对比度。

![image48](https://docs.opencv.org/4.0.0/histogram_equalization.png)

建议阅读直方图均衡的维基百科页面，了解更多相关细节。 它有一个非常好的解释和解决的例子，所以在阅读之后几乎可以理解所有内容。 在这里将看到它的Numpy实现，之后看到OpenCV函数。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('wiki.jpg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
```

![image49](https://docs.opencv.org/4.0.0/histeq_numpy1.jpg)

你可以看到直方图位于更亮的区域。 我们需要全谱。 为此，我们需要一个转换函数，它将较亮区域中的输入像素映射到整个区域中的输出像素。 这就是直方图均衡所做的。

现在我们找到最小直方图值（不包括0）并应用维基页面中给出的直方图均衡化方程。 但我在这里使用了Numpy的蒙面数组概念数组。 对于掩码数组，所有操作都在非掩码元素上执行。 

```python
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
```

现在我们有了查找表，它提供了关于每个输入像素值的输出像素值的信息。我们只要应用这个变换

```python
img2 = cdf[img]
```

现在我们像以前一样计算它的直方图和cdf，结果如下所示：

![image50](https://docs.opencv.org/4.0.0/histeq_numpy2.jpg)

另一个重要特征是，即使图像是较暗的图像（而不是我们使用的更亮的图像），在均衡后我们将得到几乎与我们相同的图像。 结果，这被用作“参考工具”以使所有图像具有相同的照明条件。 这在许多情况下很有用。 例如，在面部识别中，在训练面部数据之前，将面部图像均衡化以使它们全部具有相同的照明条件。

## OpenCV中的直方图均衡

OpenCV有一个函数来执行此操作，`cv.equalizeHist()`。 它的输入只是灰度图像，输出是我们的直方图均衡图像。

下面是一个简单的代码段，显示了我们使用的相同图像的用法：

```python
img = cv.imread('wiki.jpg',0)
equ = cv.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv.imwrite('res.png',res)
```

所以现在你可以拍摄不同光线条件的不同图像，均衡它并检查结果。

当图像的直方图被限制在特定区域时，直方图均衡是好的。 在直方图覆盖大区域的强度变化较大的地方，即存在亮像素和暗像素时，它将无法正常工作。 请查看其他资源中的SOF链接。

![image51](https://docs.opencv.org/4.0.0/equalization_opencv.jpg)

#### （3）CLAHE（对比度有限自适应直方图均衡）

我们刚看到的第一个直方图均衡，考虑了图像的全局对比度。 在许多情况下，这不是一个好主意。 例如，下图显示了全局直方图均衡后的输入图像及其结果。

![image52](https://docs.opencv.org/4.0.0/clahe_1.jpg)

确实，直方图均衡后背景对比度有所改善。但比较两个图像中的雕像的脸。由于亮度过高，我们丢失了大部分信息。这是因为它的直方图并不局限于特定区域，正如我们在之前的案例中看到的那样（尝试绘制输入图像的直方图，你将获得更多的直觉）。

因此，为了解决这个问题，使用自适应直方图均衡。在此，图像被分成称为“图块”的小块（在OpenCV中，tileSize默认为8x8）。然后像往常一样对这些块中的每一个进行直方图均衡。所以在一个小区域内，直方图会限制在一个小区域（除非有噪音）。如果有噪音，它会被放大。为避免这种情况，应用对比度限制。如果任何直方图区间高于指定的对比度限制（在OpenCV中默认为40），则在应用直方图均衡之前，将这些像素剪切并均匀分布到其他区间。均衡后，为了去除图块边框中的瑕疵，应用双线性插值。

下面的代码片段显示了如何在OpenCV中应用CLAHE：

```python
import numpy as np
import cv2 as cv
img = cv.imread('tsukuba_l.png',0)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv.imwrite('clahe_2.jpg',cl1)
```

查看下面的结果并将其与上面的结果进行比较，尤其是雕像区域：

![image53](https://docs.opencv.org/4.0.0/clahe_2.jpg)
