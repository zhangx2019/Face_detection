# -*- coding: utf-8 -*-

import cv2
import sys
from PIL import Image

def CatchUsbVideo(window_name, camera_idx):
    '''
    API详解：原型：void nameWindow(const string& winname,int flags = WINDOW_AUTOSIZE) ;
            参数1：新建的窗口的名称。自己随便取。
            参数2：窗口的标识，一般默认为WINDOW_AUTOSIZE 。
	        WINDOW_AUTOSIZE 窗口大小自动适应图片大小，并且不可手动更改。
	        WINDOW_NORMAL 用户可以改变这个窗口大小
	        WINDOW_OPENGL 窗口创建的时候会支持OpenGL
    '''
    cv2.namedWindow(window_name)

    #视频来源，可以是已存好的视频，也可以直接来自USB摄像头
    # cap = cv2.VideoCapture('D:\output.avi') 打开视频文件
    cap = cv2.VideoCapture(camera_idx)

    #告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("./config/haarcascade_frontalface_alt2.xml")

    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    #cap.isOpened()查看初始化摄像头或者打开视频文件是否成功，若返回值是True，则表明成功，否则返回值是False
    while cap.isOpened():
        #cap.read() 按帧读取视频，它的返回值有两个：ret, frame
        #其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False
        #frame就是每一帧的图像，是个三维矩阵
        ok, frame = cap.read() #读取一帧数据
        if not ok:
            break

        #将当前帧转换成灰度图像
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        '''
        classfier.detectMultiScale()即是完成实际人脸识别工作的函数，该函数参数说明如下：
        grey：要识别的图像数据（即使不转换成灰度也能识别，但是灰度图可以降低计算强度，
        因为检测的依据是哈尔特征，转换后每个点的RGB数据变成了一维的灰度，这样计算强度就减少很多）;
        scaleFactor：图像缩放比例，可以理解为同一个物体与相机距离不同，其大小亦不同，
        必须将其缩放到一定大小才方便识别，该参数指定每次缩放的比例;
        minNeighbors：对特征检测点周边多少有效点同时检测，这样可避免因选取的特征检测点太小而导致遗漏
        minSize：特征检测点的最小值
        '''
        #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        '''
        对同一个画面有可能出现多张人脸，因此，需要用一个for循环将所有检测到的人脸都读取出来，
        然后逐个用矩形框框出来，这就是接下来的for语句的作用；Opencv会给出每张人脸在图像中的起始坐标（左上角，x、y）
        以及长、宽（h、w），据此可以截取出人脸；cv2.rectangle()完成画框的工作，外扩了10个像素以框出比人脸稍大一点的区域；
        cv2.rectangle()函数的最后两个参数一个用于指定矩形边框的颜色，一个用于指定矩形边框线条的粗细程度。
        '''
        if len(faceRects) > 0:            #大于0则检测到人脸
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10),color, 2)

        #显示图像
        #cv2.imshow('iframe', gray) 播放视频，第一个参数是视频播放窗口的名称，第二个参数是视频的当前帧
        cv2.imshow(window_name, frame)
        #cv2.waitKey(10) 每一帧的播放时间，毫秒级
        c = cv2.waitKey(10)
        #停止捕获视频的方式：10毫秒或10毫秒内键盘输入了“q”,则停止捕获视频
        if c & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #1234.py a b c d e f
    #sys.argv[:]，['1234.py','a', 'b', 'c', 'd', 'e', 'f']
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        CatchUsbVideo('face recognition', 0)
