## 目标：
- 学习Meanshift和Camshift算法来寻找和跟踪视频中的物体。

## Meanshift
Meanshift 算法的基本原理很简单。假设有一堆点（比如直方图反向投影得到的点），和一个小的窗口（可能是一个圆形窗口），然后将这个窗口移动到最大像素密度处（点最多的地方）。如图所示：
![image1](https://docs.opencv.org/4.0.0/meanshift_basics.jpg)

初始窗口是蓝色的“C1”，它的圆心为蓝色方框“C1_o”，而窗口中所有点质心却是“C1_r”(小的蓝色圆圈)，很明显圆心和点的质心没有重合。所以移动圆心 C1_o 到质心 C1_r，这样我们就得到了一个新的窗口。这时又可以找到新窗口内所有点的质心，大多数情况下还是不重合的，所以重复上面的操作：将新窗口的中心移动到新的质心。就这样不停的迭代操作直到窗口的中心和其所包含点的质心重合为止（或者有一点小误差）。按照这样的操作我们的窗口最终会落在像素值（和）最大的地方。如上图所示“C2”是窗口的最后位址，我们可以看出来这个窗口中的像素点最多。下图演示了整个过程：
![image2](https://docs.opencv.org/4.0.0/meanshift_face.gif)

所以我们通常传递直方图反向投影图像和初始目标位置。当物体运动时，运动明显地反映在直方图的反向投影图像中。因此，meanshift算法将窗口移动到具有最大密度的新位置。

## OpenCV 中的 Meanshift
要在 OpenCV 中使用 Meanshift 算法，首先我们需要设置目标，找到它的直方图，这样我们就可以在每一帧上对目标进行反向投影来计算平均位移。另外我们还需要提供窗口的起始位置。对于直方图，我们仅考虑Hue（色调）值，此外，为了避免因光线不足而产生错误值，使用`cv.inRange ( )`函数将这些值忽略掉。

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('slow.flv')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv.imshow('img2',img2)

        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv.destroyAllWindows()
cap.release()
```

结果的三帧如下所示：
![image3](https://docs.opencv.org/4.0.0/meanshift_result.jpg)

## Camshift
这里面有一个问题，我们的窗口的大小是固定的，而汽车由远及近（在视觉上）是一个逐渐变大的过程，固定窗口是不合适的。所以我们需要根据目标的大小和角度来对窗口进行调整。1988年，OpenCVLabs 提出了一个解决方案：CAMshift （Continuously Adaptive Meanshift）算法，由Gary Bradsky发表在他的论文“Computer Vision Face Tracking for Use in a Perceptual User Interface”中。

Camshift算法首先应用meanshift。一旦meanshift收敛，它就会更新窗口的大小，$$ s = 2 \times \sqrt{\frac{M_{00}}{256}} $$。它还计算最佳拟合椭圆的方向。同样，它将新的缩放搜索窗口和先前的窗口位置应用于meanshift。继续该过程直到满足所需的准确度。

![image4](https://docs.opencv.org/4.0.0/camshift_face.gif)

## OpenCV 中的 Camshift

它与meanshift几乎相同，但它返回一个旋转的矩形（这是我们的结果）和box参数（用于在下一次迭代中作为搜索窗口传递）。请参阅以下代码：


```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('slow.flv')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # apply meanshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        
        # Draw it on image
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
        cv.imshow('img2',img2)
        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k)+".jpg",img2)
    
    else:
        break

cv.destroyAllWindows()
cap.release()
```

结果的三个框架如下所示：
![image5](https://docs.opencv.org/4.0.0/camshift_result.jpg)
