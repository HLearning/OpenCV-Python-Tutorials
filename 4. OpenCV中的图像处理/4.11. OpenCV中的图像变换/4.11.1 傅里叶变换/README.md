## 目标：

本章节你需要学习以下内容:

    *使用OpenCV查找图像的傅立叶变换
    *使用Numpy中提供的FFT函数
    *傅立叶变换的一些应用
    *我们将看到以下函数：cv.dft（），cv.idft（）等
    
### 1、理论

傅立叶变换用于分析各种滤波器的频率特性。对于图像，2D离散傅里叶变换（DFT）用于找到频域。称为快速傅里叶变换（FFT）的快速算法用于计算DFT。有关这些的详细信息可以在任何图像处理或信号处理教科书中找到。请参阅其他资源部分。

对于正弦信号，x（t）= Asin（2πft），我们可以说f是信号的频率，如果采用其频域，我们可以看到f处的尖峰。如果对信号进行采样以形成离散信号，则我们得到相同的频域，但在[-π，π]或[0,2π]（或对于N点DFT的[0，N]）范围内是周期性的。你可以将图像视为在两个方向上采样的信号。因此，在X和Y方向上进行傅里叶变换可以得到图像的频率表示。

更直观地说，对于正弦信号，如果幅度在短时间内变化如此之快，则可以说它是高频信号。如果变化缓慢，则为低频信号。你可以将相同的想法扩展到图像。幅度在图像中的幅度变化很大？在边缘点，或噪音。我们可以说，边缘和噪声是图像中的高频内容。如果幅度没有太大变化，则它是低频分量。 （一些链接被添加到Additional Resources_，它通过示例直观地解释了频率变换）。

现在我们将看到如何找到傅立叶变换。

### 2、Numpy中的傅里叶变换

首先，我们将看到如何使用Numpy找到傅立叶变换。Numpy有一个FFT包来做到这一点。np.fft.fft2（）为我们提供了一个复杂数组的频率变换。它的第一个参数是输入图像，它是灰度。第二个参数是可选的，它决定了输出数组的大小。如果它大于输入图像的大小，则在计算FFT之前用零填充输入图像。如果小于输入图像，则将裁剪输入图像。如果没有传递参数，则输出数组大小将与输入相同。

现在，一旦得到结果，零频率分量（DC分量）将位于左上角。如果要将其置于中心位置，则需要在两个方向上将结果移动$\frac{N}{2}$。这只是通过函数np.fft.fftshift（）完成的。 （分析更容易）。找到频率变换后，你可以找到幅度谱。

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('messi5.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image57](https://docs.opencv.org/4.0.0/fft1.jpg)

请注意，你可以在中心看到更多更白的区域，显示低频内容更多。

所以你找到了频率变换现在你可以在频域做一些操作，比如高通滤波和重建图像，即找到逆DFT。 为此，你只需通过使用尺寸为60x60的矩形窗口进行遮罩来移除低频。 然后使用np.fft.ifftshift（）应用反向移位，以便DC组件再次出现在左上角。 然后使用np.ifft2（）函数找到逆FFT。 结果再次是一个复杂的数字。 你可以采取它的绝对价值。

```python
rows, cols = img.shape
crow,ccol = rows/2 , cols/2
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image58](https://docs.opencv.org/4.0.0/fft2.jpg)

结果显示高通滤波是边缘检测操作。这是我们在Image Gradients章节中看到的。这也表明大多数图像数据存在于光谱的低频区域。无论如何，我们已经看到如何在Numpy中找到DFT，IDFT等。现在让我们看看如何在OpenCV中完成它。

如果你仔细观察结果，特别是JET颜色的最后一个图像，你可以看到一些文物（我用红色箭头标记的一个实例）。它在那里显示出一些类似波纹的结构，它被称为振铃效应。它是由我们用于遮蔽的矩形窗口引起的。此蒙版转换为sinc形状，这会导致此问题。因此矩形窗口不用于过滤。更好的选择是高斯Windows。

### 3、OpenCV中的傅里叶变换

OpenCV为此提供了cv.dft（）和cv.idft（）函数。它返回与之前相同的结果，但有两个通道。第一个通道将具有结果的实部，第二个通道将具有结果的虚部。输入图像应首先转换为np.float32。我们将看到如何做到这一点。

```python
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('messi5.jpg',0)
dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
```

**注意：你还可以使用cv.cartToPolar（），它可以一次性返回幅度和相位**

所以，现在我们必须进行逆DFT。 在之前的会话中，我们创建了一个HPF，这次我们将看到如何去除图像中的高频内容，即我们将LPF应用于图像。 它实际上模糊了图像。 为此，我们首先在低频处创建具有高值（1）的掩模，即我们传递LF内容，并且在HF区域传递0。

```python
rows, cols = img.shape
crow,ccol = rows/2 , cols/2
# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image59](https://docs.opencv.org/4.0.0/fft4.jpg)

> 注意：像往常一样，OpenCV函数cv.dft（）和cv.idft（）比Numpy函数更快。 但是Numpy功能更加用户友好。 有关性能问题的更多详细信息，请参阅以下部分。

### 4、DFT的性能优化

对于某些阵列大小，DFT计算的性能更好。 当阵列大小为2的幂时，它是最快的。 尺寸为2，3和5的乘积的阵列也可以非常有效地处理。 因此，如果你担心代码的性能，可以在找到DFT之前将数组的大小修改为任何最佳大小（通过填充零）。 对于OpenCV，你必须手动填充零。 但对于Numpy，你可以指定FFT计算的新大小，它会自动为你填充零。

那么我们如何找到这个最佳尺寸？ OpenCV为此提供了一个函数cv.getOptimalDFTSize（）。 它适用于cv.dft（）和np.fft.fft2（）。 让我们使用IPython magic命令timeit检查它们的性能。

```python
In [16]: img = cv.imread('messi5.jpg',0)
In [17]: rows,cols = img.shape
In [18]: print("{} {}".format(rows,cols))
342 548
In [19]: nrows = cv.getOptimalDFTSize(rows)
In [20]: ncols = cv.getOptimalDFTSize(cols)
In [21]: print("{} {}".format(nrows,ncols))
360 576
```

看，大小（342,548）被修改为（360,576）。 现在让我们用零填充它（对于OpenCV）并找到它们的DFT计算性能。 你可以通过创建一个新的大零数组并将数据复制到它，或使用cv.copyMakeBorder（）来实现。

```python
nimg = np.zeros((nrows,ncols))
nimg[:rows,:cols] = img
```

或者

```python
right = ncols - cols
bottom = nrows - rows
bordertype = cv.BORDER_CONSTANT #just to avoid line breakup in PDF file
nimg = cv.copyMakeBorder(img,0,bottom,0,right,bordertype, value = 0)
```

现在我们计算Numpy函数的DFT性能比较：

```python
In [22]: %timeit fft1 = np.fft.fft2(img)
10 loops, best of 3: 40.9 ms per loop
In [23]: %timeit fft2 = np.fft.fft2(img,[nrows,ncols])
100 loops, best of 3: 10.4 ms per loop
```

它显示了4倍的加速。现在我们将尝试使用OpenCV函数。

```python
In [24]: %timeit dft1= cv.dft(np.float32(img),flags=cv.DFT_COMPLEX_OUTPUT)
100 loops, best of 3: 13.5 ms per loop
In [27]: %timeit dft2= cv.dft(np.float32(nimg),flags=cv.DFT_COMPLEX_OUTPUT)
100 loops, best of 3: 3.11 ms per loop
```

它还显示了4倍的加速。 你还可以看到OpenCV函数比Numpy函数快3倍。这也可以进行逆FFT测试，这可以作为练习。

### 4、为什么拉普拉斯算子是高通滤波器？

在论坛中提出了类似的问题。 问题是，为什么拉普拉斯算子是高通滤波器？ 为什么Sobel是HPF？ 第一个答案就是傅立叶变换。 只需将拉普拉斯算子的傅里叶变换用于更高尺寸的FFT。 分析一下：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# simple averaging filter without scaling parameter
mean_filter = np.ones((3,3))
# creating a gaussian filter
x = cv.getGaussianKernel(5,10)
gaussian = x*x.T
# different edge detecting filters
# scharr in x-direction
scharr = np.array([[-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3]])
# sobel in x direction
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# sobel in y direction
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])
# laplacian
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])
filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sobel_y', 'scharr_x']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]
for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
plt.show()
```

窗口将如下图显示：

![image60](https://docs.opencv.org/4.0.0/fft5.jpg)
