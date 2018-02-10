#!/usr/bin/python
# -- coding: utf-8 --
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to use dlib's face recognition tool.  This tool maps
#   an image of a human face to a 128 dimensional vector space where images of
#   the same person are near to each other and images from different people are
#   far apart.  Therefore, you can perform face recognition by mapping faces to
#   the 128D space and then checking if their Euclidean distance is small
#   enough.
#
#   When using a distance threshold of 0.6, the dlib model obtains an accuracy
#   of 99.38% on the standard LFW face recognition benchmark, which is
#   comparable to other state-of-the-art methods for face recognition as of
#   February 2017. This accuracy means that, when presented with a pair of face
#   images, the tool will correctly identify if the pair belongs to the same
#   person or is from different people 99.38% of the time.
#
#   Finally, for an in-depth discussion of how dlib's tool works you should
#   refer to the C++ example program dnn_face_recognition_ex.cpp and the
#   attendant documentation referenced therein.
#
#
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  This code will also use CUDA if you have CUDA and cuDNN
#   installed.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.
#   操作方法：把照片放在/home/zzy/face_recognition_image/下 分别建立文件夹（可以是正常的照片不必为脸部），修改name=[]数组对应人的姓名，test.py中也需要修改
#   修改person_num=4 一共几个人在训练库中
import sys
import os
import dlib
import glob
from skimage import io
import csv
import numpy as np

predictor_path = "/home/zzy/dlib-19.6/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "/home/zzy/dlib-19.6/dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "/home/zzy/face_recognition_image/"

out = open("/home/zzy/face_recognition_image/test.csv","w")
writer = csv.writer(out,dialect='excel')
name=["zzy","szc","zh","jxn"]
# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
win = dlib.image_window()
k=0
count=1
person_num=4
# Now process all the images

for k in range(0,person_num):
    for f in glob.glob(os.path.join(faces_folder_path+name[k], "*.png")):
        img = io.imread(f)
        win.clear_overlay()
        win.set_image(img)
        dets = detector(img, 1)

        print("Number of faces detected: {}".format(len(dets)))
        shape = sp(img, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        print(face_descriptor)
        face_descriptor=np.insert(face_descriptor, 128, values=np.uint8(k), axis=0) #128-rows axis=cols label!
        face_descriptor_trans=np.transpose(face_descriptor)
        writer.writerow(face_descriptor_trans)
        # count=count+1
        # if count%5==0:
        #     k=k+1

