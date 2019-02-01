## 目标：
- 学习将图像从一个颜色空间转换为另一个颜色空间，例如BGR↔Gray，BGR↔HSV
- 创建一个提取视频中某个特定彩色对象的应用程序
- 学习以下函数：`cv.cvtColor()`，`cv.inRange()`
    
## 改变色彩空间
OpenCV中有150多种颜色空间转换方法。我们只研究两种最广泛使用的转换方法，BGR↔Gray和BGR↔HSV。
对于颜色转换，使用函数`cv.cvtColor(input_image，flag)`，其中flag确定转换类型。
对于BGR→Gray转换，我们使用标志`cv.COLOR_BGR2GRAY`。类似地，对于BGR→HSV，我们使用标志`cv.COLOR_BGR2HSV`。要获取其他标志，只需在Python终端中运行以下命令：

```python
>>> import cv2 as cv
>>> flags = [i for i in dir(cv) if i.startswith('COLOR_')]
>>> print( flags )
```

> 注意：对于HSV色彩空间，色调的取值范围是[0,179]，饱和度的取值范围是[0,255]，明度的取值范围是[0,255]。不同的软件可能使用不同的取值方式，因此，如果要将OpenCV的HSV值与其他软件的HSV值进行比较时，则需要对这些范围进行标准化。

## 对象提取
现在我们知道如何将BGR图像转换为HSV，我们可以使用HSV色彩空间来提取彩色对象。在HSV中表示颜色比在BGR颜色空间中更容易。在我们的程序中，我们将尝试提取视频画面中的蓝色对象。下面是方法程序执行步骤：

- 获取视频中的每一帧
- 从BGR转换为HSV颜色空间
- 我们为HSV图像设定一系列的蓝色阈值
- 单独提取蓝色对象并显示，之后我们便可以对我们想要的图像做任何事情。

以下是详细评论的代码：

```python
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while(1):
    
    # Take each frame
    _, frame = cap.read()
    
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv.destroyAllWindows()
```

下面的图片展示了我们提取蓝色对象后的效果：

![image1](https://docs.opencv.org/4.0.0/frame.jpg)

> 注意：图像中有一些噪音，我们将在后面的章节中看到如何删除它们。这是对象提取中最简单的方法。一旦你学习了轮廓的功能，你就可以做很多事情，比如找到这个物体的重心并用它来追踪物体，只需在镜头前移动你的手以及许多其他有趣的东西来绘制图表。

## 如何去查找确定HSV值
这是我们在stackoverflow.com中常见的问题。其实解决这个问题非常简单，你可以使用相同的函数cv.cvtColor()。你只需传递所需的BGR值，而不是传递图像。例如，要查找绿色的HSV值，在Python终端中输入以下命令：

```python
>>> green = np.uint8([[[0,255,0 ]]])
>>> hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
>>> print( hsv_green )
[[[ 60 255 255]]]
```

现在分别将[H-10,100,100]和[H+10,255,255]作为下限和上限。除了这种方法，你可以使用任何图像编辑工具如GIMP，或任何在线转换器来查找这些值，但不要忘记调整HSV范围。