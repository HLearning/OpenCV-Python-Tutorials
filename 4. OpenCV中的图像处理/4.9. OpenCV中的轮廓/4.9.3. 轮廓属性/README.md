## 目标：
- 学习提取一些常用的对象属性，如Solidity，Equivalent Diameter，Mask image，Mean Intensity

## 1. Aspect Ratio（长宽比）

它是对象的边界矩形的宽度与高度的比。

$$ Aspect\ Ratio= \frac{Width}{Height} $$

```python
x,y,w,h = cv.boundingRect(cnt)
aspect_ratio = float(w)/h
```

## 2. Extent（大小比）

它是轮廓区域与边界矩形区域的比。

$$ Extent= \frac{Object\ Area}{Bounding\ Rectangle\ Area} $$

```python
area = cv.contourArea(cnt)
x,y,w,h = cv.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area
```

## 3. Solidity（密实比）

Solidity是轮廓区域与其凸包区域的比率。

$$ Solidity= \frac{Contour\ Area}{Convex\ Hull\ Area} $$

```python
area = cv.contourArea(cnt)
hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area)/hull_area
```

## 4. Equivalent Diameter（等效直径）

等效直径是圆的直径，其面积与轮廓面积相同。

$$ Equivalent\ Diameter=\sqrt{\frac{4\times Contour\ Area}{\pi }} $$

```python
area = cv.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)
```

## 5. Orientation（方向）

方向是对象定向的角度。以下方法还给出了主轴和短轴长度。

```python
(x,y),(MA,ma),angle = cv.fitEllipse(cnt)
```

## 6. Mask & Pixel Points（掩模和像素点）

在某些情况下，我们可能需要包含该对象的所有点。它可以如下完成：

```python
mask = np.zeros(imgray.shape,np.uint8)
cv.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv.findNonZero(mask)
```

这里，两个方法，一个使用Numpy函数，另一个使用OpenCV函数（最后一个注释行）给出相同的方法。 结果也相同，但略有不同。 Numpy以**（行，列）**格式给出坐标，而OpenCV以**（x，y）**格式给出坐标。所以答案基本上会互换。请注意，row=x和column=y。

## 7. 最大值，最小值及其位置

我们可以使用掩模图像找到这些参数。

```python
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(imgray,mask = mask)
```

## 8. 平均颜色或平均灰度

在这里，我们可以找到对象的平均颜色。或者它可以是灰度模式下物体的平均强度。我们再次使用相同的面具来做到这一点。

```python
mean_val = cv.mean(im,mask = mask)
```

## 9. 极点

极值点表示对象的最顶部，最底部，最右侧和最左侧的点。

```python
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
```

例如，如果我将它应用于印度地图，我会得到以下结果：

![image38](https://docs.opencv.org/4.0.0/extremepoints.jpg)
