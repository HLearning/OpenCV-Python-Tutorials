## 目标：
- 在本小节你将学习把滑动条绑定到OpenCV窗口中
- 你将学习以下几个函数：`cv.getTrackbarPos()`, `cv.createTrackbar()`

## 代码示例
在这里，我们将创建一个简单的应用程序，完成显示指定的颜色。你有一个显示颜色的窗口和三个滑动条，分别用于指定B，G，R各颜色。你可以去拖动滑动条上的按钮去更改窗口所显示的颜色。默认情况下，初始颜色将设置为黑色。
对于`cv.getTrackbarPos()`函数，第一个参数是滑动条名称，第二个参数是它所附加的窗口名称，第三个参数是默认值，第四个参数是最大值，第五个参数是执行的回调函数每次轨迹栏值都会发生变化。回调函数始终具有默认参数，即滑动条位置。在我们的例子中，函数什么都不做，所以我们简单地跳过。
轨迹栏的另一个重要应用是将其用作按钮或开关。默认情况下，OpenCV没有按钮功能。因此，你可以使用滑动条来获得此类功能。在我们的应用程序中，我们创建了一个开关，其中应用程序仅在开关打开时有效，否则屏幕始终为黑色。
参考一下代码：

```python
import numpy as np
import cv2 as cv

def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')

# create trackbars for color change
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image',0,1,nothing)

while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv.getTrackbarPos('R','image')
    g = cv.getTrackbarPos('G','image')
    b = cv.getTrackbarPos('B','image')
    s = cv.getTrackbarPos(switch,'image')
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

cv.destroyAllWindows()
```

窗口将如下图所示：

![image4](https://docs.opencv.org/4.0.0/trackbar_screenshot.jpg)

