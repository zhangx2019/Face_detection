# -*-coding:utf8-*-

import os
import cv2
import time
import shutil

# 将目录下的所有文件进行列表
def getAllPath(dirpath, *suffix):
    PathArray = []
    # os.walk(dirpath,dirnames,filenames)中第一个为起始路径，第二个为起始路径下的文件夹，第三个是起始路径下的文件
    # os.path.split（）返回文件的路径和文件名
    # dirname,filename=os.path.split('/home/ubuntu/python_coding/split_func/split_function.py')
    # print dirname
    # print filename
    # #输出为：# /home/ubuntu/python_coding/split_func #split_function.py
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            '''
            #os.path.splitext()将文件名和扩展名分开
            fname,fename=os.path.splitext('/home/ubuntu/python_coding/split_func/split_function.py')
            print 'fname is:',fnameprint 'fename is:',fename
            #输出为：# fname is:/home/ubuntu/python_coding/split_func/split_function
                     #fename is:.py
            '''
            # # os.path.splitext() 返回的是一个元祖，取最后一个文件后缀名用[-1]
            #     suffix = os.path.splitext(pic_urls[i])[-1]
            if os.path.splitext(fn)[1] in suffix:
                '''
                #os.path.join() 将分离的部分合成一个整体
                filename=os.path.join('/home/ubuntu/python_coding','split_func')
                print filename
                #输出为：/home/ubuntu/python_coding/split_func
                '''
                fname = os.path.join(r, fn)
                PathArray.append(fname)
    return PathArray

# 截取符合要求的人脸图片
def readPicSaveFace_1(sourcePath, targetPath, invalidPath, *suffix):
    num=0
    try:
        ImagePaths = getAllPath(sourcePath, *suffix)

        # 对list中图片逐一进行检查,找出其中的人脸然后写到目标文件夹下
        count = 1
        # haarcascade_frontalface_alt.xml为库训练好的分类器文件，下载opencv，安装目录中可找到
        face_cascade = cv2.CascadeClassifier('./config/haarcascade_frontalface_alt.xml')
        for imagePath in ImagePaths:
            try:
                img = cv2.imread(imagePath) #读入图片

                if type(img) != str:
                    # 1.1和5分别为图片缩放比例和需要检测的有效点数
                    faces = face_cascade.detectMultiScale(img, 1.1, 5)
                    if len(faces):
                        for (x, y, w, h) in faces:
                            # 设置人脸宽度大于128像素，去除较小的人脸
                            if w >= 128 and h >= 128:
                                # 扩大图片，可根据坐标调整
                                X = int(x)
                                W = min(int(x + w), img.shape[1])
                                Y = int(y)
                                H = min(int(y + h), img.shape[0])

                                f = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
                                cv2.imwrite(targetPath + r'\\' + str(num) + '.jpg', f)#保存图片
                                num += 1
                                count += 1
                                print(imagePath + "have face")
                    # else:
                    #   shutil.move(imagePath, invalidPath)
            except:
                continue
    except IOError:
        print("Error")
    else:
        print('Find ' + str(count - 1) + ' faces to Destination ' + targetPath)


if __name__ == '__main__':
    invalidPath = r'./data/invalid/'
    sourcePath = r'./data/telangpu/'
    targetPath1 = r'./data/others/'
    readPicSaveFace_1(sourcePath, targetPath1, invalidPath, '.jpg', '.JPG', 'png', 'PNG')
