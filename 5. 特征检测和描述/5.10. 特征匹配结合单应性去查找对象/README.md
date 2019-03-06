## 目标：
- 我们将联合使用特征提取和 calib3d 模块中的 findHomography 在复杂图像中查找已知对象。
    
## 基础

还记得上一节我们做了什么吗？我们使用一个查询图像，在其中找到一些特征点（关键点），我们又在另一幅图像中也找到了一些特征点，最后对这两幅图像之间的特征点进行匹配。简单来说就是：我们在一张杂乱的图像中找到了一个对象（的某些部分）的位置。这些信息足以帮助我们在目标图像中准确的找到（查询图像）对象。

为了达到这个目的我们可以使用 calib3d 模块中的 cv2.findHomography()函数。如果将这两幅图像中的特征点集传给这个函数，他就会找到这个对象的透视图变换。然后我们就可以使用函数 cv2.perspectiveTransform() 找到这个对象了。至少要 4 个正确的点才能找到这种变换。

我们已经知道在匹配过程可能会有一些错误，而这些错误会影响最终结果。为了解决这个问题，算法使用 RANSAC 和 LEAST_MEDIAN(可以通过参数来设定)。所以好的匹配提供的正确的估计被称为 inliers，剩下的被称为outliers。cv2.findHomography() 返回一个掩模，这个掩模确定了 inlier 和outlier 点。

让我们来搞定它吧！

## 代码实现

首先，像往常一样，让我们在图像中找到SIFT特征并应用比率测试来找到最佳匹配。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv.imread('box.png',0)          # queryImage
img2 = cv.imread('box_in_scene.png',0) # trainImage

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
```

现在我们设置一个条件，即至少10个匹配（由MIN_MATCH_COUNT定义）才去查找目标。否则只是显示一条消息，说明没有足够的匹配。

如果找到了足够的匹配，我们要提取两幅图像中匹配点的坐标。把它们传入到函数中计算透视变换。一旦我们找到 3x3 的变换矩阵，就可以使用它将查询图像的四个顶点（四个角）变换到目标图像中去了。然后再绘制出来。

```python
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    h,w,d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
```

最后，我们绘制内部函数（如果成功找到对象）或匹配关键点（如果失败）。

```python
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()
```

请参阅下面的结果。 对象在杂乱图像中标记为白色：

![image24](https://docs.opencv.org/4.0.0/homography_findobj.jpg)