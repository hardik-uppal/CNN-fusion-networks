# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:29:41 2020

@author: hardi
"""

import os
from shutil import copyfile
from average_face import extract_points, average_face

import cv2


CurtinFaces = 'D:/CurtinFaces_crop/RGB/test1/'
iiitd_rgbd = 'D:/RGB_D_Dataset_new/fold1/test/RGB/'
lock3dface = ''
#############

CurtinFaces_out = 'D:/TEST_FOLDER/average_face/CurtinFaces'
iiitd_rgbd_out = 'D:/TEST_FOLDER/average_face/IIITD_RGBD'
lock3dface_out = 'D:/TEST_FOLDER/average_face/Lock3dface'

for subject in os.listdir(CurtinFaces):
    subject_path = os.path.join(CurtinFaces,subject)
    for image in os.listdir(subject_path):
        if image == '04.jpg':
            source = os.path.join(subject_path, image)
            image_dest = str(subject) +'_'+ image 
            destination = os.path.join(CurtinFaces_out, image_dest)
            copyfile(source,destination)
#        break
    
predictor_path = 'D:/tutorial/AE-Gan_reperesentation/shape_predictor_68_face_landmarks.dat'


extract_points(predictor_path, CurtinFaces_out)
average_face(CurtinFaces_out)



for subject in os.listdir(iiitd_rgbd):
    subject_path = os.path.join(iiitd_rgbd,subject)
    for image in os.listdir(subject_path):
#        if image == '04.jpg':
        source = os.path.join(subject_path, image)
        image_dest = str(subject) +'_'+ image 
        destination = os.path.join(iiitd_rgbd_out, image_dest)
        copyfile(source,destination)
        break

extract_points(predictor_path, iiitd_rgbd_out)
average_face(iiitd_rgbd_out)

iiitd_rgbd_out_cam = 'D:/TEST_FOLDER/average_face/IIITD_RGBD_cam_only_rgb'

for image in os.listdir(iiitd_rgbd_out_cam):
    path = os.path.join(iiitd_rgbd_out_cam,image)
    image_cam = cv2.imread(path,0)
    norm_image = cv2.normalize(image_cam, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cam_colormap = cv2.applyColorMap(cv2.convertScaleAbs(norm_image, alpha=0.09), cv2.COLORMAP_JET)
#        image_depth = cv2.cvtColor(image_depth, cv2.COLOR_BGR2RGB)
    cv2.imwrite('D:/TEST_FOLDER/average_face/test_cam_norm.jpg',cam_colormap)
    break