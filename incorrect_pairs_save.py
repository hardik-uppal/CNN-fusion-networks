# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 05:01:58 2020

@author: hardi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()

#file_path = 'D:/TEST_FOLDER/lock3d/Incorrect_examples'


def save_pairs(img, ref_img, file_path, cat, db):
    '''
    img: is list pred class, actual class, pred class image, actual class image 
    ref_img : dict of refrence images for a class
    cat : string correct pair or inccorect pair
    db: db name, string
    '''
#    for img in incorrects.items():
            
    fig, axes = plt.subplots(1,3,figsize=(14,5))
#    print(len(img))
#    print(img[2].shape)
#    plt.axis('off')
#    plt.grid(b=None)
    axes[0].imshow(img[2][0][0])
    axes[0].grid(False)
    axes[0].axis(False)
    axes[1].imshow(img[2][1][0])
    axes[1].grid(False)
    axes[1].axis(False)
    axes[2].imshow(ref_img[img[0][0]][0])
    axes[2].grid(False)
    axes[2].axis(False)

    img_path = file_path  +'/{}_{}_{}.png'.format(img[0][0],img[1][0],cat)
    fig.savefig(img_path)
    plt.close()
    

def get_pairs(test_generator_1, model, file_path, db):
    
    incorrects= []
    corrects= []
#    ref_img = {}
#    i=0
    ref_img = get_ref_img(test_generator_1)
    for input_x_y in tqdm(test_generator_1(1)):
#        print(input_x_y)
        
        predictions = model.predict_on_batch(input_x_y[0])
        pred_classes = list(np.argmax(predictions, axis=1))
        act_classes = list(np.argmax(input_x_y[1], axis=1))
#        i=i+1
        if (pred_classes != act_classes):
            incorrects = [pred_classes,act_classes,input_x_y[0]]
#            break
            save_pairs(incorrects, ref_img, file_path, 'incorrects', db)
#        if (pred_classes == act_classes):
#            corrects = [pred_classes,act_classes,input_x_y[0]]
#            save_pairs(corrects, ref_img, file_path, 'corrects', db)
#            ref_img[act_classes[0]] = input_x_y[0][0]
    return incorrects, corrects


def get_ref_img(test_generator_2):
    

    ref_img = {}
#    i=0
    for input_x_y in test_generator_2(1):
     
        act_classes = list(np.argmax(input_x_y[1], axis=1))    
        ref_img[act_classes[0]] = input_x_y[0][0]
    return ref_img
