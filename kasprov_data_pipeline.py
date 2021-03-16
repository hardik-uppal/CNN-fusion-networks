# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:59:14 2020

@author: hardi
"""

import numpy as np 
import os
import random
from keras.utils.np_utils import to_categorical
import cv2

from scipy.io import loadmat

#D:\RGBDI_Dataset_manual_FaceDetection_ksprov\1\Kinect2\Video1\RGB

class data_pipeline:
    
    def __init__(self, image_dir, dataset, test_train=None, test_train_split=None):
        self.image_full_arr = []
        self.label_full_arr = []
#        self.batch_size = batch_size
        self.dataset = dataset
        self.test_train = test_train
        self.test_train_split = test_train_split
        #check if input dir is list or not; create two lsit with addresses and corresponding labels
        if isinstance(image_dir, list):
            for image_dir_temp in image_dir:
                label_list = os.listdir(image_dir_temp)
                
                for label in label_list:
                    label_dir = os.path.join(image_dir_temp,label)
                    for image in os.listdir(label_dir):
                        if image[-3:] in ['jpg','bmp','png']:
                            image_path = os.path.join(label_dir,image)
                            self.image_full_arr.append(image_path)
                            self.label_full_arr.append(label)
                        
        else:
            label_list = os.listdir(image_dir)
            
            if dataset == 'kasprov':
                
                for label in label_list:
                    label_dir = image_dir +'/' +str(label)
                    label_dir_temp1 = label_dir + '/Kinect2/Video'
                    for session in range(1,3):
                        label_dir_temp2 = label_dir_temp1 + str(session) +'/RGB/'
                        for image in os.listdir(label_dir_temp2):
                            if image[-3:] in ['jpg','bmp','png']:
                                image_path = os.path.join(label_dir_temp2,image)
                                self.image_full_arr.append(image_path)
                                self.label_full_arr.append(label)
#                                image_full_arr.append(image_path)
#                                label_full_arr.append(label)
            
            else:
                for label in label_list:
                    label_dir = os.path.join(image_dir,label)
                    for image in os.listdir(label_dir):
                        if image[-3:] in ['jpg','bmp','png']:
                            image_path = os.path.join(label_dir,image)
                            self.image_full_arr.append(image_path)
                            self.label_full_arr.append(label)
            
        
        if self.test_train:
            ## join both list for filtering
            arr_joined = np.column_stack((self.image_full_arr,self.label_full_arr))
#            arr_joined = np.column_stack((image_full_arr,label_full_arr))
            # if test_train exists, set the filter for test labels
            if self.dataset == 'pandora':
                filter = np.asarray(['10.0','14.0','16.0','20.0'])
            elif self.dataset == 'IIITD':
                filter = np.asarray(['10.0','14.0','16.0','20.0'])
            elif self.dataset == 'curtinfaces':
                filter = np.asarray(['10','14','16','20'])
            elif self.dataset == 'eurecom':
                filter = np.asarray(['10.0','14.0','16.0','20.0'])
            else:
                filter = None
            
#            print(arr_joined[:1])
#            divide data for each label in case of kasprov
               
            if dataset =='kasprov':
                if self.test_train_split:
                    
#                    test_arr, train_arr = [],[]
                    
                    test_arr_temp, train_arr_temp = [], []
                    for label_class in set(self.label_full_arr):
                        temp_class_arr = arr_joined[np.in1d(arr_joined[:, 1], label_class)]
                        _test_idx = np.random.choice(temp_class_arr.shape[0], int(temp_class_arr.shape[0]*self.test_train_split), replace=False)
                        _train_idx = [idx for idx in range(temp_class_arr.shape[0]) if idx not in _test_idx]
                        test_arr_temp.extend(np.take(temp_class_arr, _test_idx,axis=0))
                        train_arr_temp.extend(np.take(temp_class_arr, _train_idx,axis=0))
        #                    test_arr.extend(test_arr_temp)
        #                    train_arr.extend(train_arr_temp)
                    test_arr = np.asarray(test_arr_temp)
                    train_arr = np.asarray(train_arr_temp)
                
            else:
                test_arr = arr_joined[np.in1d(arr_joined[:, 1], filter)]
                train_arr = arr_joined[np.in1d(arr_joined[:, 1], filter, invert=True)]
            
            self.image_train_arr = train_arr[:,0]
            label_train_arr = train_arr[:,1].astype(np.float)
            self.image_test_arr = test_arr[:,0]
            label_test_arr = test_arr[:,1].astype(np.float)
            
            # change label to categorical
            self.label_train_arr_onehot = to_categorical(label_train_arr)
            #find ids which are zero for all labels
            idx_train = np.argwhere(np.all(self.label_train_arr_onehot[..., :] == 0, axis=0))
            #remove id with zero labels
            self.label_train_arr_onehot = np.delete(self.label_train_arr_onehot, idx_train, axis=1)
            
            #same operation for test data
            self.label_test_arr_onehot = to_categorical(label_test_arr)
            idx_test = np.argwhere(np.all(self.label_test_arr_onehot[..., :] == 0, axis=0))
            self.label_test_arr_onehot = np.delete(self.label_test_arr_onehot, idx_test, axis=1)            

#            self.steps_per_epochs = len(image_full_arr)/batch_size
            self.num_classes_train = len(set(label_train_arr))
            self.num_classes_test = len(set(label_test_arr))  
            if test_train:
                print('Train:\n')
                print('{} Images found in {} classes'.format(len(self.image_train_arr),self.num_classes_train))
#            elif test_train=='test':
                print('Test:\n')
                print('{} Images found in {} classes'.format(len(self.image_test_arr),self.num_classes_test))
        else:
            self.label_full_arr_onehot = to_categorical(self.label_full_arr)
            idx_full = np.argwhere(np.all(self.label_full_arr_onehot[..., :] == 0, axis=0))
            self.label_full_arr_onehot = np.delete(self.label_test_arr_onehot, idx_full, axis=1) 

            self.num_classes_full = len(set(self.label_full_arr))
            
   
            
    def gen_batch(self, X1, X2, batch_size):
    
        while True:
            idx = np.random.choice(X1.shape[0], batch_size, replace=False)
            yield X1[idx], X2[idx]            
            
    def normalization(self, X):
        result = X / 127.5 - 1
        
        # Deal with the case where float multiplication gives an out of range result (eg 1.000001)
        out_of_bounds_high = (result > 1.)
        out_of_bounds_low = (result < -1.)
#        out_of_bounds = out_of_bounds_high + out_of_bounds_low
        
        if not all(np.isclose(result[out_of_bounds_high],1)):
#            print(result)
            result[out_of_bounds_high] = 1
#            raise RuntimeError("Normalization gave a value greater than 1")
        else:
            result[out_of_bounds_high] = 1.
            
        if not all(np.isclose(result[out_of_bounds_low],-1)):
#            print(result)
            result[out_of_bounds_low] = -1
#            raise RuntimeError("Normalization gave a value lower than -1")
        else:
            result[out_of_bounds_low] = -1.
        
        return result       
            
            
    def flow_from_dir(self, batch_size, target_dim, test_train):
        self.test_train = test_train
        
        if self.test_train == 'train':
            image_iter_arr = self.image_train_arr
            label_iter_arr = self.label_train_arr_onehot
        elif self.test_train == 'test':
            image_iter_arr = self.image_test_arr
            label_iter_arr = self.label_test_arr_onehot
        else:
            image_iter_arr = self.image_full_arr
            label_iter_arr = self.label_full_arr_onehot

        # iterate over the list of image and label batch
        for batch in self.gen_batch(image_iter_arr, label_iter_arr, batch_size):
                
            image_arr_batch = []
            image_depth_arr_batch = []
#            print(batch)
            try:
                image_path_batch = batch[0]
                #load label and convert to one hot
                y_batch = batch[1]
                
                #read and resize image to target_dim
                for image_path in image_path_batch:
                    
        #            print(image_depth_path)
                    if self.dataset == 'eurecom':
                        image_depth_path = image_path.replace('/RGB/','/depth/')
                        image_depth_path = image_depth_path.replace('rgb_','depth_')  
        #                print(image_depth_path)
                    elif self.dataset == 'pandora':
                        image_depth_path = image_path.replace('_RGB/','_depth/')
                        image_depth_path = image_depth_path.replace('_rgb.','_depth.')
                    elif self.dataset == 'kasprov':
                        image_depth_path = image_path.replace('/RGB/','/Depth/')
                        image_depth_path = image_depth_path.replace('.png','.mat')
                    else:
                        image_depth_path = image_path.replace('/RGB/','/depth/')
                    
                    
                    #read rgb and depth image and add to batch
                    image_arr = cv2.imread(image_path)
#                    print('read {}'.format(image_arr.shape))
                    image_arr = cv2.resize(image_arr,target_dim[:-1])
#                    print('resize {}'.format(image_arr.shape))
                    image_arr_batch.append(image_arr)
                    
                    #read depth images
                    if self.dataset=='kasprov':
#                    
                        image_depth_arr = loadmat(image_depth_path)
                        image_depth_arr = image_depth_arr['depthFaceMat']
                        image_depth_arr = cv2.normalize(image_depth_arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        
                    else:
                        image_depth_arr = cv2.imread(image_depth_path,0)
                    
                    image_depth_arr = cv2.resize(image_depth_arr,target_dim[:-1])
#                    image_depth_arr = np.expand_dims(image_depth_arr, axis=-1)
                    image_depth_arr_batch.append(image_depth_arr)     

            except:
#                print(image_depth_arr.shape)
                print('datagen failed for {} and {}'.format(image_path,image_depth_path))
                break          
    
    #                    
            x_batch_norm = self.normalization(np.asarray(image_arr_batch))
            x_batch_depth_norm = self.normalization(np.asarray(image_depth_arr_batch))
            x_batch_depth_norm = np.expand_dims(x_batch_depth_norm, axis=-1)
            yield x_batch_norm, x_batch_depth_norm, y_batch
    