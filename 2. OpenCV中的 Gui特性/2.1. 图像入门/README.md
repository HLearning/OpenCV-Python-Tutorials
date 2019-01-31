## 目标：
- 学会如何读取、显示以及保存图像。
- 学习函数：`cv.imread()`, `cv.imshow() `, `cv.imwrite()`
- 用Matplotlib显示图像

## OpenCV的使用
## 读取图像
`cv.imread()`函数用于读取图像，需要注意的是，该图像应该处在Python代码源文件或者已给出完整路径的工作目录中。针对函数的第二个参数，通过以下几个例子分别说明各自功能。
- cv.IMREAD_COLOR：默认参数，以彩色模式加载图像，图像的透明度将被忽略。
- cv.IMREAD_GRAYSCALE：以灰度模式加载图像。
- cv.IMREAD_UNCHANGED：以alpha通道模式加载图像。

> 注意：你也可以通过传递1，0，-1来代替上面三个函数功能。

参考代码：

```python
import numpy as np
import cv2 as cv
# Load an color image in grayscale
img = cv.imread('messi5.jpg',0)
```

> 注意：即使图像路径错误，它也不会抛出任何错误，但是`print(img)`会显示`None`

## 显示图像
`cv.imshow()`函数被用于在窗口中显示图像，窗口会自动适应图像大小。
其中，函数的第一个参数是窗口的名称，是字符串类型。第二个参数是要加载的图像。你可以显示多个图像窗口，但是每个窗口名称必须不同。

参考代码：

```python
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
```

窗口如图所示：

![image1](https://docs.opencv.org/4.0.0/opencv_screenshot.jpg)

`cv.waitKey()`是一个键盘事件函数，它的参数以毫秒为单位，该函数在毫秒的时间内去等待键盘事件，如果时间之内有键盘事件触发则程序继续，如果函数参数设置为0，则无限时间的等待键盘事件触发。它也可以设置为检测指定按键的触发，比如等待按键a的触发，我们将在下面讨论。

> 注意：这个函数除了可以等待键盘事件的触发之外还可以处理很多其他的GUI事件，所以你必须把它放在显示图像函数之后。

`cv.destroyAllWindow()`函数用于关闭我们所创建的所有显示图像的窗口，如果想要关闭特定的窗口，请使用cv.destroyWindow()函数，把要关闭的窗口名称作为参数。

> 注意：特别地，你也可以先创建一个窗口，再加载图像。在这种情况下，你可以使用`cv.nameWindow()`函数自行调整窗口大小。函数默认参数是`cv.WINDOW_AUTOSIZE`。你可以使用`cv.WINDOW_NORMAL`函数，以自行调整窗口大小。当图像尺寸太大或者需要添加滚动条时， 上述函数将会非常有用。

参考以下代码：

```python
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
```

## 保存图像
`cv.imwrite()`函数用于保存图像。其中第一个参数是保存为的图片名，第二个参数为待保存图像。
参考一下代码：
```python
cv.imwrite('messigray.png',img)
```
上述代码将图片保存为PNG格式。

## 总结
下面的代码程序将加载并显示为灰度图像，按's'则保存图像并退出，按'ESC'键直接退出且不保存。

参考代码：

```python
import numpy as np
import cv2 as cv
img = cv.imread('messi5.jpg',0)
cv.imshow('image',img)
k = cv.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv.imwrite('messigray.png',img)
    cv.destroyAllWindows()
```

> 注意：如果你使用的计算机是64位，用`k = cv.waitKey(0)`代替：`k = cv.waitKey(0) ＆ 0xFF`

## 使用Matplotlib
Matplotlib是Python的绘图库，为你提供各种绘图方法。  接下来，你将学习如何使用Matplotlib显示图像、缩放图片和保存图片等。下面将详述其使用方法。
参考代码：
```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('messi5.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
```

窗口如图所示：

![image2](https://docs.opencv.org/4.0.0/matplotlib_screenshot.jpg)

Matplotlib提供了大量的绘图选项。有关更多详细信息，请参阅Matplotlib文档。 在以后的学习中，我们将详细介绍。

> 注意：OpenCV加载的彩色图像处于BGR模式，但Matplotlib以RGB模式显示。因此，如果使用OpenCV读取图像，则Matplotlib中的彩色图像将无法正确显示。请参阅练习了解更多详情。