## 目标：
在图像处理中，由于每秒需要处理大量操作，因此处理图像的代码必须不仅要能给出正确的结果，同时还必须要快。所以在本小节中，你将学习：
- 衡量代码的性能。
- 一些优化代码性能的技巧。
- 你将学习以下几个函数：`cv.getTickCount`, `cv.getTickFrequency`

除了OpenCV库之外，Python还提供了一个time模块，有助于测量执行时间。另一个profile模块可以获得有关代码的详细报告，例如代码中每个函数所花费的时间，调用函数的次数等。如果你使用的是IPython，所有这些功能都以一个有好的方式整合到一起。

## 使用 OpenCV 衡量程序效率
`cv.getTickCount`函数返回参考事件（如机器开启时刻）到调用此函数的时钟周期数。因此，如果在函数执行之前和之后都调用它，则会获得用于执行函数的时钟周期数。

`cv.getTickFrequency`函数返回时钟周期的频率，或每秒钟的时钟周期数。所以，要想获得函数的执行时间，你可以执行以下操作：

```python
e1 = cv.getTickCount()
# your code execution
e2 = cv.getTickCount()
time = (e2 - e1)/ cv.getTickFrequency()
```

我们将展示以下示例示例，下面的例子使用从5到49几个不同大小的核进行中值滤波。（不要考虑结果会是什么样的，这不是我们的目的）：

```python
img1 = cv.imread('messi5.jpg')

e1 = cv.getTickCount()
for i in xrange(5,49,2):
    img1 = cv.medianBlur(img1,i)
e2 = cv.getTickCount()
t = (e2 - e1)/cv.getTickFrequency()
print( t )

# Result I got is 0.521107655 seconds
```

> 注意：你可以使用time模块的函数执行相同操作来替代cv.getTickCount，使用time.time()函数,然后取两次结果的时间差。

## OpenCV中的默认优化
许多OpenCV的功能都使用SSE2，AVX等进行了优化。它还包含了一些未经优化的代码。因此，如果我们的系统支持这些功能，我们应该利用它们（几乎所有现代处理器都支持它们），编译时是默认启用优化。因此，OpenCV运行的代码就是已优化代码（如果已启用），否则运行未优化代码。 你可以使用`cv.useOptimized()`来检查它是否已启用/禁用优化，并使用cv.setUseOptimized())来启用/禁用它。让我们看一个简单的例子。

```python
# check if optimization is enabled
In [5]: cv.useOptimized()
Out[5]: True

In [6]: %timeit res = cv.medianBlur(img,49)
10 loops, best of 3: 34.9 ms per loop

# Disable it
In [7]: cv.setUseOptimized(False)
In [8]: cv.useOptimized()
Out[8]: False

In [9]: %timeit res = cv.medianBlur(img,49)
10 loops, best of 3: 64.1 ms per loop
```

优化的中值滤波比未优化的版本快2倍。如果检查其来源，你可以看到中值滤波是经过SIMD优化的。因此，你可以使用它来在代码顶部启用优化（请记住它默认启用）。

## 在IPython中的进行性能性能

有时您可能需要比较两个类似操作的性能。IPython为执行此操作提供了一个神奇的命令timeit。它多次运行代码以获得更准确的结果。同样，它们也适用于测量单行代码。
例如，你知道下面哪个加法运算更好，x = 5;y = x**2, x = 5;y = x*x, x = np.uint8([5]);y = x*x or y = np.square(x)?我们将在IPython shell中使用timeit得到答案。

```python
In [10]: x = 5
In [11]: %timeit y=x**2
10000000 loops, best of 3: 73 ns per loop

In [12]: %timeit y=x*x
10000000 loops, best of 3: 58.3 ns per loop

In [15]: z = np.uint8([5])
In [17]: %timeit y=z*z
1000000 loops, best of 3: 1.25 us per loop

In [19]: %timeit y=np.square(z)
1000000 loops, best of 3: 1.16 us per loop
```

你可以看到，x = 5; y = x * x是最快的，与Numpy相比快了约20倍。如果你也考虑创建矩阵，它可能会快达100倍。很酷对不对 （Numpy开发者正在研究这个问题）

>注意：Python标量操作比Numpy标量操作更快。因此对于包含一个或两个元素的操作，Python标量优于Numpy数组。当阵列的大小稍大时，Numpy会占据优势。

我们将再尝试一个例子。 这次，我们将比较同一图像的`cv.countNonZero()`和`np.count_nonzero()`的性能。
```python
In [35]: %timeit z = cv.countNonZero(img)
100000 loops, best of 3: 15.8 us per loop
In [36]: %timeit z = np.count_nonzero(img)
1000 loops, best of 3: 370 us per loop
```

你可以看到，OpenCV的执行性能比Numpy快将近25倍。

> 注意：通常，OpenCV函数比Numpy函数更快。因此对于相同的操作，OpenCV功能是首选。但是可能也有例外，尤其是当使用Numpy对视图而不是复制数组时。

## 更多的IPython命令
还有几个魔法命令可以用来检测程序的执行效率，profiling，line profiling，memory measurement等。他们都有完善的文档。所以这里只提供了超链接。感兴趣的读者可以自己学习一下。

## 性能优化技术
有几种技术和编码方法可以利用Python和Numpy的最大性能。此处仅注明相关的内容，并提供重要来源的链接。这里要注意的主要是，首先尝试以简单的方式实现算法。一旦工作，对其进行分析，找到瓶颈并进行优化。

1. 尽量避免在Python中使用循环，尤其是双层/三层嵌套循环等。它们本身就很慢。
2. 将算法/代码尽量使用向量化操作，因为Numpy和OpenCV针对向量运算进行了优化。
3. 利用高速缓存一致性。
4. 除非需要，否则不要复制数组。尝试使用视图去替代复制数组。数组复制是一项非常浪费资源的操作。

即使在完成所有这些操作之后，如果你的代码仍然很慢，或者使用大型循环是不可避免的，请使用其他库（如Cython）来加快速度。
