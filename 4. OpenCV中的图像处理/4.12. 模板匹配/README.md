## 目标：

- 使用模板匹配查找图像中的对象
- 函数：`cv.matchTemplate()`，`cv.minMaxLoc()`
    
## 理论

模板匹配是一种在较大图像中搜索和查找模板图像位置的方法。为此，OpenCV附带了一个函数cv.matchTemplate（）。它只是在输入图像上滑动模板图像（如在2D卷积中），并比较模板图像下的输入图像的模板和补丁。在OpenCV中实现了几种比较方法。 （你可以查看文档以获取更多详细信息）。它返回一个灰度图像，其中每个像素表示该像素的邻域与模板匹配的程度。

如果输入图像的大小（WxH）且模板图像的大小（wxh），则输出图像的大小为（W-w + 1，H-h + 1）。获得结果后，可以使用cv.minMaxLoc（）函数查找最大/最小值的位置。将其作为矩形的左上角，并将（w，h）作为矩形的宽度和高度。那个矩形是你的模板区域。

> 注意：如果你使用cv.TM_SQDIFF作为比较方法，则最小值会给出最佳匹配。

## OpenCV中的模板匹配

在这里，作为一个例子，我们将在他的照片中搜索梅西的脸。所以我创建了一个模板如下：

![image61](https://docs.opencv.org/4.0.0/messi_face.jpg)

我们将尝试所有比较方法，以便我们可以看到它们的结果如何：

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('messi5.jpg',0)
img2 = img.copy()
template = cv.imread('template.jpg',0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
```

请参阅以下结果：

* cv.TM_CCOEFF

![image62](https://docs.opencv.org/4.0.0/template_ccoeff_1.jpg)

* cv.TM_CCOEFF_NORMED

![image63](https://docs.opencv.org/4.0.0/template_ccoeffn_2.jpg)

* cv.TM_CCORR

![image64](https://docs.opencv.org/4.0.0/template_ccorr_3.jpg)

* cv.TM_CCOEFF_NORMED

![image65](https://docs.opencv.org/4.0.0/template_ccorrn_4.jpg)

* cv.TM_SQDIFF

![image66](https://docs.opencv.org/4.0.0/template_sqdiff_5.jpg)

* cv.TM_SQDIFF_NORMED

![image67](https://docs.opencv.org/4.0.0/template_sqdiffn_6.jpg)

你可以看到使用cv.TM_CCORR的结果不如我们预期的那样好。

## 与多个对象匹配的模板

在上一节中，我们搜索了Messi脸部的图像，该图像仅在图像中出现一次。 假设你正在搜索多次出现的对象，cv.minMaxLoc（）将不会为你提供所有位置。 在这种情况下，我们将使用阈值。 所以在这个例子中，我们将使用着名游戏Mario的截图，我们将在其中找到硬币。

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img_rgb = cv.imread('mario.png')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('mario_coin.png',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imwrite('res.png',img_rgb)
```

窗口将如下图显示：

![image68](https://docs.opencv.org/4.0.0/res_mario.jpg)