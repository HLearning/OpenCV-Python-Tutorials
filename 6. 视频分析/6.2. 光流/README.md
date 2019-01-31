## 目标：
- 了解光流的概念及 Lucas-Kanade 光流法。
- 使用`cv.calcOpticalFlowPyrLK()`函数来跟踪视频中的特征点。

## 光流
由于目标对象或者摄像机的移动造成的图像对象在连续两帧图像中的移动被称为光流。它是一个 2D 向量场，可以用来表示一个点从第一帧图像到第二帧图像之间的移动。如下图所示。
![image6](https://docs.opencv.org/4.0.0/optical_flow_basic1.jpg)

上图显示了一个球在连续的五帧图像间的移动。箭头表示其位移向量。光流在很多领域中都很有用：
- 由运动重建结构
- 视频压缩
- 视频防抖

光流是基于以下假设下工作的：
1. 在连续的两帧图像之间，目标对象的像素的灰度值不改变。
2. 相邻像素具有相似的运动。

考虑第一帧中的像素 $$ I(x,y,t) $$，它在dt时间之后移动距离$$ (dx,dy) $$。根据第一条假设：灰度值不变。所以我们可以得到：

$$ I(x,y,t) = I(x+dx, y+dy, t+dt) $$

然后对等号右侧采用泰勒级数展开，删除相同项并两边除以dt得到以下等式：

$$ f_x u + f_y v + f_t = 0 \; $$

其中：

$$ f_x = \frac{\partial f}{\partial x} \; ; \; f_y = \frac{\partial f}{\partial y} $$

$$ u = \frac{dx}{dt} \; ; \; v = \frac{dy}{dt} $$

上边的公式叫做光流方程。其中 $$ f_x $$ 和 $$ f_y $$ 是图像梯度，同样 $$ f_t $$ 是时间方向的梯度。但（u，v）是不知道的。我们不能在一个等式中求解两个未知数，有几个方法可以帮我们解决这个问题，其中的一个是 Lucas-Kanade 法。

## Lucas-Kanade方法
现在我们要使用第二条假设，邻域内的所有点都有相似的运动。LucasKanade 法就是利用一个 3x3 邻域中的 9 个点具有相同运动的这一点。这样我们就可以找到$$ (f_x, f_y, f_t) $$这 9 个点的光流方程，用它们组成一个具有两个未知数 9 个等式的方程组，这是一个约束条件过多的方程组。一个好的解决方法就是使用最小二乘拟合。下面就是求解结果：

$$ \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} \sum_{i}{f_{x_i}}^2 & \sum_{i}{f_{x_i} f_{y_i} } \\ \sum_{i}{f_{x_i} f_{y_i}} & \sum_{i}{f_{y_i}}^2 \end{bmatrix}^{-1} \begin{bmatrix} - \sum_{i}{f_{x_i} f_{t_i}} \\ - \sum_{i}{f_{y_i} f_{t_i}} \end{bmatrix} $$

（你会发现上边的逆矩阵与 Harris 角点检测器非常相似，这说明角点很适合被用来做跟踪）
从使用者的角度来看，想法很简单，我们去跟踪一些点，然后我们就会获得这些点的光流向量。但是还有一些问题。直到现在我们处理的都是很小的运动。如果有大的运动怎么办呢？图像金字塔。当我们进入金字塔时，小运动被移除，大运动变成小运动。因此，通过在那里应用Lucas-Kanade，我们就会得到尺度空间上的光流。

## Lucas-KanadeOpenCV中的Lucas-Kanade光流
上述所有过程都被 OpenCV 打包成了一个函数`cv2.calcOpticalFlowPyrLK()`。现在我们使用这个函数创建一个小程序来跟踪视频中的一些点。我们使用函数 `cv2.goodFeatureToTrack() `来确定要跟踪的点。我们首先在视频的第一帧图像中检测一些 Shi-Tomasi 角点，然后我们使用 LucasKanade 算法迭代跟踪这些角点。我们要给函数` cv2.calcOpticlaFlowPyrLK()`传入前一帧图像和其中的点，以及下一帧图像。函数将返回带有状态数的点，如果状态数是 1，那说明在下一帧图像中找到了这个点（上一帧中角点），如果状态数是 0，就说明没有在下一帧图像中找到这个点。我们再把这些点作为参数传给函数，如此迭代下去实现跟踪。代码如下：
```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('slow.flv')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv.destroyAllWindows()
cap.release()
```

(上面的代码没有对返回角点的正确性进行检查。图像中的一些特征点甚至在丢失以后，光流还会找到一个预期相似的点。所以为了实现稳定的跟踪，我们应该每个一定间隔就要进行一次角点检测。OpenCV 的官方示例中带有这样一个例子，它是每 5 帧进行一个特征点检测。它还对光流点使用反向检测来选取好的点进行跟踪。示例为/samples/python2/lk_track.py)

结果如下图所示：
![image7](https://docs.opencv.org/4.0.0/opticalflow_lk.jpg)

## OpenCV中的密集光流
Lucas-Kanade 法是计算稀疏特征集的光流（上面的例子使用Shi-Tomasi 算法检测角点）。OpenCV 还提供了一种计算稠密光流的方法，它会计算图像中的所有点的光流。这是基于 Gunner_Farneback 的算法，2003年Gunner Farneback在“Two-Frame Motion Estimation Based on Polynomial Expansion”中对该算法进行了解释。
下面的例子就是使用上面的算法计算稠密光流。结果是一个带有光流向量（u，v）的双通道数组。通过计算我们能得到光流的大小和方向。我们使用颜色对结果进行编码以便于更好的观察。方向对应于 H（Hue）通道，大小对应于 V（Value）通道。代码如下：

```python
import cv2 as cv
import numpy as np

cap = cv.VideoCapture("vtest.avi")

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next

cap.release()
cv.destroyAllWindows()
```

请看下面的结果：
![image8](https://docs.opencv.org/4.0.0/opticalfb.jpg)

OpenCV 的官方示例中有一个更高级的稠密光流算法，具体请参阅/samples/python2/opt_flow.py。
