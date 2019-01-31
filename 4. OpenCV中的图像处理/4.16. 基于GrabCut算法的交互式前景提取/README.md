## 目标：

本章节你需要学习以下内容:

- 我们将看到GrabCut算法来提取图像中的前景
- 我们将为此创建一个交互式应用程序。

## 理论

GrabCut算法由英国剑桥微软研究院的Carsten Rother，Vladimir Kolmogorov和Andrew Blake设计。在他们的论文中，“GrabCut”：使用迭代图切割的交互式前景提取。前景提取需要一种算法，用户交互最少，结果就是GrabCut。

从用户的角度来看它是如何工作的？最初用户在前景区域周围绘制一个矩形（前景区域应该完全在矩形内）。然后算法迭代地对其进行分段以获得最佳结果。完成。但在某些情况下，分割将不会很好，例如，它可能已将某些前景区域标记为背景，反之亦然。在这种情况下，用户需要进行精细的修饰。只需对图像进行一些描述，其中存在一些错误结果。笔划基本上说*“嘿，这个区域应该是前景，你标记它的背景，在下一次迭代中纠正它”*或它的背景相反。然后在下一次迭代中，你将获得更好的结果。

见下图。第一名球员和足球被包围在一个蓝色矩形中。然后进行一些具有白色笔划（表示前景）和黑色笔划（表示背景）的最终修饰。我们得到了一个很好的结果。

![image82](https://docs.opencv.org/4.0.0/grabcut_output1.jpg)

那么背景会发生什么？

* 用户输入矩形。这个矩形之外的所有东西都将被视为确定的背景（这就是之前提到的矩形应包括所有对象的原因）。矩形内的一切都是未知的。类似地，任何指定前景和背景的用户输入都被视为硬标签，这意味着它们不会在过程中发生变化。
* 计算机根据我们提供的数据进行初始标记。它标记前景和背景像素（或硬标签）
* 现在，高斯混合模型（GMM）用于模拟前景和背景。
* 根据我们提供的数据，GMM学习并创建新的像素分布。也就是说，未知像素被标记为可能的前景或可能的背景，这取决于其在颜色统计方面与其他硬标记像素的关系（它就像聚类一样）。
* 从该像素分布构建图形。图中的节点是像素。添加了另外两个节点，Source节点和Sink节点。每个前景像素都连接到Source节点，每个背景像素都连接到Sink节点。
* 将像素连接到源节点/端节点的边的权重由像素是前景/背景的概率来定义。像素之间的权重由边缘信息或像素相似性定义。如果像素颜色存在较大差异，则它们之间的边缘将获得较低的权重。
* 然后使用mincut算法来分割图形。它将图形切割成两个分离源节点和汇聚节点，具有最小的成本函数。成本函数是被切割边缘的所有权重的总和。切割后，连接到Source节点的所有像素都变为前景，连接到Sink节点的像素变为背景。
* 该过程一直持续到分类收敛为止。

如下图所示（图片提供：http：//www.cs.ru.ac.za/research/g02m1682/）

![image83](https://docs.opencv.org/4.0.0/grabcut_scheme.jpg)

## 示例

现在我们使用OpenCV进行抓取算法。 OpenCV具有此功能，cv.grabCut（）。我们将首先看到它的论点：

* img - 输入图像
* mask - 这是一个掩码图像，我们指定哪些区域是背景，前景或可能的背景/前景等。它由以下标志cv.GC_BGD，cv.GC_FGD，cv.GC_PR_BGD，cv.GC_PR_FGD完成，或者只是通过图像0,1,2,3。
* rect - 矩形的坐标，包括格式为（x，y，w，h）的前景对象
* bdgModel，fgdModel - 这些是内部算法使用的数组。你只需创建两个大小为（n = 1.65）的np.float64类型零数组。
* iterCount - 算法应运行的迭代次数。
* mode - 它应该是cv.GC_INIT_WITH_RECT或cv.GC_INIT_WITH_MASK或组合，它决定我们是绘制矩形还是最终的触摸笔画。

首先让我们看看矩形模式。我们加载图像，创建一个类似的蒙版图像。我们创建了fgdModel和bgdModel。我们给出矩形参数。这一切都是直截了当的。让算法运行5次迭代。模式应该是cv.GC_INIT_WITH_RECT，因为我们使用矩形。然后运行抓取。它修改了蒙版图像。在新的掩模图像中，像素将被标记为表示背景/前景的四个标记，如上所述。因此，我们修改掩模，使得所有0像素和2像素都被置为0（即背景），并且所有1像素和3像素被置为1（即前景像素）。现在我们的最后面具准备好了。只需将其与输入图像相乘即可得到分割后的图像。

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('messi5.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,450,290)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
```

窗口将如下图显示：

![image84](https://docs.opencv.org/4.0.0/grabcut_rect.jpg)

哎呀，梅西的头发不见了。 没有头发谁喜欢梅西？ 我们需要把它带回来。 因此，我们将为其提供1像素（确定前景）的精细修饰。 与此同时，有些地方已经出现了我们不想要的图片，还有一些标识。 我们需要删除它们。 在那里我们提供一些0像素的修饰（确定背景）。 因此，正如我们现在所说的那样，我们在之前的案

我实际上做的是，我在绘图应用程序中打开输入图像，并在图像中添加了另一层。 在画中使用画笔工具，我在这个新图层上标记了带有黑色的白色和不需要的背景（如徽标，地面等）的前景（头发，鞋子，球等）。 然后用灰色填充剩余的背景。 然后在OpenCV中加载该掩模图像，编辑我们在新添加的掩模图像中使用相应值的原始掩模图像。 检查以下代码：

```python
# newmask is the mask image I manually labelled
newmask = cv.imread('newmask.png',0)
# wherever it is marked white (sure foreground), change mask=1
# wherever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
```
窗口将如下图显示：
![image85](https://docs.opencv.org/4.0.0/grabcut_mask.jpg)