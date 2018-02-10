# -- coding: utf-8 --
#   操作方法：把照片放在/home/zzy/face_recognition_image/下 分别建立文件夹（可以是正常的照片），修改name=[]数组对应人的姓名
#   修改video_capture = cv2.VideoCapture("/home/zzy/yawn_data/szc.avi")测试视频还是实时检测

import sys
import os
import dlib
import glob
from skimage import io
import csv
import numpy as np
import cv2
import pyttsx
from datetime import datetime
import socket


address=('192.168.1.107',8085)   #set the address
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
order=["zzy","szc","zh","jxn","zjw","zrj","gyq","Unknown"]

def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataMat=[];labelMat=[]
    dataset = list(lines)
    # for i in range(len(dataset)):
    #     dataset[i] = [float(x) for x in dataset[i]]
    for i in range(len(dataset)):
        lineArr=[]
        vector = dataset[i]
        labelMat.append(float(vector[-1]))
        for j in range(0,128):
            lineArr.append(float(vector[j]))
        dataMat.append(lineArr)
    return dataMat,labelMat


def cos(vector1,vector2):
    dot_product = 0.0;
    normA = 0.0;
    normB = 0.0;
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)


if __name__ == '__main__':
    engine = pyttsx.init()

    [dataMat,labelMat]=loadCsv("/home/zzy/face_recognition_image/test.csv")
    predictor_path = "/home/zzy/dlib-19.6/shape_predictor_68_face_landmarks.dat"
    face_rec_model_path = "/home/zzy/dlib-19.6/dlib_face_recognition_resnet_model_v1.dat"
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    total_image=np.shape(labelMat)[0]
    name=["zzy","szc","zh","jxn","zjw","zrj","gyq","Unknown"]
    #     用视频测试
    #video_capture = cv2.VideoCapture("/home/zzy/yawn_data/fyp.avi")
    #video_capture = cv2.VideoCapture("/home/zzy/jilu_record/gyq/gyq.avi")
    video_capture = cv2.VideoCapture(0)
    while (True):
        starttime = datetime.now()
        ret, frame = video_capture.read()  # Read the image with OpenCV
        #print(frame[:,:,0])
        #cv2.equalizeHist(frame[],frame)
        dets = detector(frame, 1)
        # if dets is None:
        #     raise Exception("Unable to align the frame")
        for k, d in enumerate(dets):
            shape = sp(frame, dets[0])
        name_index=-1
        try:
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            face_descriptor_trans = np.transpose(face_descriptor)
            dist=np.zeros(total_image)
            for i in range(0,total_image):
                dist[i]=1-cos(face_descriptor,dataMat[i])
            dist1=dist.tolist()
            dist_min=dist.min()
            print(dist_min)
            posit_min = dist1.index(dist_min)
            name_index = np.uint8(labelMat[posit_min])
            print(name[name_index])
            if dist_min>0.035:   #控制陌生人
                name_index=-1
            else:
                posit_min=dist1.index(dist_min)
                print(posit_min)
                name_index = np.uint8(labelMat[posit_min])
            print(name[name_index])

            s.sendto(order[name_index], address)

            engine.say("You are" + str(name[name_index]))
            endtime = datetime.now()
            once_time=(endtime - starttime).total_seconds()
            print(once_time)
            #break
        except:
            print('no person!')
        cv2.putText(frame,"Hello,"+str(name[name_index]+"!") , (400, 30), cv2.FONT_HERSHEY_SIMPLEX,
    1.0, (0, 255, 0), 2)
        cv2.imshow("face_recog", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    #     用图片测试
    # frame=cv2.imread("/home/zzy/face_recognition_image/test_photo/58.png")
    # height=np.shape(frame)[0]
    # width=np.shape(frame)[1]
    # dets = detector(frame, 1)
    # for k, d in enumerate(dets):
    #     shape = sp(frame, dets[0])
    # name_index = -1
    # try:
    #     face_descriptor = facerec.compute_face_descriptor(frame, shape)
    #     face_descriptor_trans = np.transpose(face_descriptor)
    #     dist = np.zeros(total_image)
    #     for i in range(0, total_image):
    #         dist[i] = 1 - cos(face_descriptor, dataMat[i])
    #     dist1 = dist.tolist()
    #     dist_min = dist.min()
    #     print(dist_min)
    #     if dist_min > 0.05:  # 控制陌生人
    #         name_index = -1
    #     else:
    #         posit_min = dist1.index(dist_min)
    #         print(posit_min)
    #         name_index = np.uint8(labelMat[posit_min])
    #     print(name[name_index])
    # except:
    #     print('no face in the picture!')
    # cv2.putText(frame,  str(name[name_index]), (height/2, width/2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # cv2.imshow("face_recog", frame)
    # cv2.waitKey(0)