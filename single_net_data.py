#!/usr/bin/python
# -- coding: utf-8 --
import sys
import os
import dlib
import glob
from skimage import io
import csv
import numpy as np

predictor_path = "/home/zzy/dlib-19.6/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "/home/zzy/dlib-19.6/dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "/home/zzy/face_recognition_image/gyq"

out = open("/home/zzy/face_recognition_image/test1.csv","w")
writer = csv.writer(out,dialect='excel')
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

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        img = io.imread(f)
        win.clear_overlay()
        win.set_image(img)
        dets = detector(img, 1)

        print("Number of faces detected: {}".format(len(dets)))
        shape = sp(img, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        print(face_descriptor)
        face_descriptor=np.insert(face_descriptor, 128, values=np.uint8(6), axis=0) #128-rows axis=cols label!
        face_descriptor_trans=np.transpose(face_descriptor)
        writer.writerow(face_descriptor_trans)