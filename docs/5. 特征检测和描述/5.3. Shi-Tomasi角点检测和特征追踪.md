## 目标：
- 了解另一个角点探测器：Shi-Tomasi角点探测器
- 函数：`cv.goodFeaturesToTrack()`
    
## 理论

在上一小节，我们看到了Harris角点检测。1994年晚些时候，J.Shi和C.Tomasi在他们的论文《Good Features to Track》中做了一个小修改，与Harris角点检测相比显示出更好的结果。Harris角点探测器的评分功能由下式给出：

$$R = \lambda_1 \lambda_2 - k(\lambda_1+\lambda_2)^2$$

除此之外，Shi-Tomasi提出：

$$R = min(\lambda_1, \lambda_2)$$

如果它大于阈值，则将其视为拐角。如果我们像在Harris角点检测器中那样在$\lambda_1 - \lambda_2$空间中绘制它，我们得到如下图像：

![image6](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image6.jpg)

从图中可以看出，只有当$$\lambda_1$$和$$\lambda_2$$高于最小值λmin时，它才被视为一个角（绿色区域）。

## 代码实现

OpenCV有一个函数`cv.goodFeaturesToTrack()`。 它通过Shi-Tomasi方法（或Harris角点检测，如果你指定它）在图像中找到N个最强角。像往常一样，图像应该是灰度图像。然后指定要查找的角点数。然后指定质量等级，该等级是0-1之间的值，表示低于每个人被拒绝的角点的最低质量。然后我们提供检测到的角之间的最小欧氏距离。

利用所有这些信息，该函数可以在图像中找到角点。低于质量水平的所有角点都被拒绝。然后它根据质量按降序对剩余的角进行排序。然后功能占据第一个最强的角点，抛弃最小距离范围内的所有角点并返回N个最强的角点。

在下面的示例中，我们将尝试找到25个最佳角点：

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('blox.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)
    
plt.imshow(img),plt.show()
```

结果如下图所示：

![image7](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/5.Feature%20Detection%20and%20Description/Image/image7.jpg)

我们以后会发现这个函数很适合在目标跟踪中使用。

