## 目标：
- 在本小节你将学习读取、显示和保存视频
- 你将学习用摄像头捕获并显示视频
- 你将学习以下几个函数：`cv.VideoCapture()`, `cv.VideoWriter()`

## 用摄像头捕获视频
通常，我们需要用摄像头来捕获直播画面，OpenCV为此提供了一些非常简单的函数接口。下面我们来尝试用摄像头来捕获视频画面（我使用的是电脑的内置摄像头）并将画面转化成灰度图像显示，这项操作很简单。
如果要捕获视频，首先要做的是创建一个VideoCapture对象，它的参数可以是设备索引或者是视频文件的名称。设备索引就是指设备所对应的设备号，当只连接一个摄像头，参数只需传递0（或-1） 。你可以传递参数1来选择你连接的第二个摄像头等等。接下来，你需要逐帧捕获并显示并不要忘记关闭捕获。

参考一下代码：

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()    

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('frame',gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
```

`cap.read()`返回一个bool值（True / False）。如果读取帧正确，则它将为True。因此，你可以通过值来确定视频的结尾。
如果初始化摄像头失败，上面的代码会报错。你可以使用`cap.isOpened()`来检查是否初始化。如果返回值是True，说明初始化成功，否则就要使用函数 `cap.open()`。

你还可以使用`cap.get(propld)`方法访问此视频的某些功能，参数propId代表0到18之间的数字。每个数字表示视频的一个属性，详细的信息参见：[cv::VideoCapture::get()](https://docs.opencv.org/4.0.0/d8/dfe/classcv_1_1VideoCapture.html#aa6480e6972ef4c00d74814ec841a2939).其中一些值可以使用`cap.set(propId，value)`进行修改。其中参数value是你想要的新值。

例如，我可以通过`cap.get(cv.CAP_PROP_FRAME_WIDTH)`和`cap.get(cv.CAP_PROP_FRAME_HEIGHT)`分别检查帧宽和高度。它返回给我默认值640x480。但如果我想将其修改为320x240，只需使用`ret=cap.set(cv.CAP_PROP_FRAME_WIDTH，32)`和`ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT，240)` 。

> 注意：如果收到报错信息，请确保其他使用摄像头的程序在正常工作（如Linux中的Cheese）。

## 播放视频文件

与从相机捕获视频原理相同，只需将设备索引更改为视频文件的名字。同时在显示帧时，请给cv.waitKey()函数传递适当的时间参数。如果它太小，视频将非常快，如果它太高，视频将会很慢。在正常情况下，25毫秒就可以了。

参考代码：
```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vtest.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame',gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```

> 注意：确保已安装正确版本的ffmpeg或gstreamer。 若使用Video Capture遇到麻烦，可能原因是错误安装了ffmpeg/gstreamer。

## 保存视频
目前为止我们可以捕获视频，并逐帧显示，现在我们希望保存该视频。保存图片很简单，但是对于视频， 相对繁琐很多。
首先创建一个VideoWriter对象，我们应该指定输出文件名（例如：output.avi），然后我们应该指定FourCC代码并传递每秒帧数（fps）和帧大小。最后一个是isColor标志，如果是True，则每一帧是彩色图像，否则每一帧是灰度图像。

FourCC是用于指定视频编解码器的4字节代码。可以在fourcc.org中找到可用代码列表，它取决于平台。以下编解码器对我来说是有用的：
- 在Fedora中：DIVX，XVID，MJPG，X264，WMV1，WMV2。（XVID更为可取.MJPG会产生高大小的视频.X264提供非常小的视频）
- 在Windows中：DIVX（更多要测试和添加）
- 在OSX中：MJPG（.mp4），DIVX（.avi），X264（.mkv）。

从相机捕获图像之后，在垂直方向上翻转每一帧之后逐帧保存。

参考代码：

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
```
