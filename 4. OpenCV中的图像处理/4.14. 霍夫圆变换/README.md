## 目标：
本章节你需要学习以下内容:
- 我们将学习使用Hough变换来查找图像中的圆圈。
- 我们将看到这些函数：cv.HoughCircles()

##  理论

圆圈在数学上表示为$\left ( x-x_{center} \right )^{2}+\left ( y-y_{center} \right )^{2}=r^{2}$其中$\left ( x_{center},y_{center} \right )$是圆的中心，r是圆的半径。从等式中，我们可以看到我们有3个参数，因此我们需要一个用于霍夫变换的3D累加器，这将非常无效。 因此，OpenCV使用更棘手的方法，Hough Gradient Method，它使用边缘的梯度信息。

我们在这里使用的函数是cv.HoughCircles（）。它有很多论据，在文档中有很好的解释。所以我们直接转到代码。

```python
import numpy as np
import cv2 as cv
img = cv.imread('opencv-logo-white.png',0)
img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()
```
窗口将如下图显示：
![image75](https://docs.opencv.org/4.0.0/houghcircles2.jpg)