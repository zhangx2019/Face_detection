# -*- coding: utf-8 -*-

import cv2
import sys

from PIL import Image

def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)

    #视频来源，可以是已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    #告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("./config/haarcascade_frontalface_alt2.xml")

    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据，frame就是每一帧的图像，是个三维矩阵
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将当前桢图像转换成灰度图像

        #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:          #大于0则检测到人脸
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect

                #将当前帧保存为图片
                img_name = '%s/%d.jpg'%(path_name, num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                # cv2.imwrite(file，img，num)保存一个图像。第一个参数是要保存的文件名，第二个参数是要保存的图像。
                # 可选的第三个参数，它针对特定的格式：对于JPEG，其表示的是图像的质量，用0 - 100的整数表示，默认95;
                # 对于png ,第三个参数表示的是压缩级别。默认为3.
                cv2.imwrite(img_name, image) #保存图像

                num += 1
                if num > (catch_pic_num):   #如果超过指定最大保存数量退出循环
                    break

                #画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                #显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(img, str(i), (123,456)), font, 2, (0,255,0), 3) 各参数依次是：
                # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)

        #超过指定最大保存数量结束程序
        if num > (catch_pic_num):break

        #显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo("catch face", 0, 100, './data/zhangxun')

