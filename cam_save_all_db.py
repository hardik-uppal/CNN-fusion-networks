# USAGE
# python apply_gradcam.py --image images/space_shuttle.jpg
# python apply_gradcam.py --image images/beagle.jpg
# python apply_gradcam.py --image images/soccer_ball.jpg --model resnet

# import the necessary packages
#from GradCam.gradcam import GradCAM
#for rgb
#from rgb_vqa_conv_attention_curtin_test import Caps_att
#from rgb_vqa_conv_attention_iiit_d_without_weights  import Caps_att

#from rgb_vqa_conv_attention_curtin_test_model_without_weights import Caps_att
#for multimodal
#from mutimodal_vqa_conv_attention_CurtinFaces_dataaug_for_cam import Caps_att
#from lock3d_cropped_attention_zhang import Caps_att
from CurtinFaces_cropped_attention_zhang import Caps_att

#from lock3d_uncropped_attention import Caps_att
#from CurtinFaces_uncropped_attention import Caps_att
#from iiitd_cropped_attention import Caps_att
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
import numpy as np
#import argparse
#import imutils
import cv2
#import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import optimizers
from keras import backend as K
import h5py
from keract import get_activations, display_heatmaps, display_activations, get_gradients_of_activations,  get_gradients_of_trainable_weights
#from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
#from cam_attempt_keract import display_heatmaps
from scipy.ndimage.interpolation import zoom
import keras
import os
from kasprov_depth_attention_tfis import Caps_att
# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to the input image")
#ap.add_argument("-m", "--model", type=str, default="vgg",
#	choices=("vgg", "resnet"),
#	help="model to be used")
#args = vars(ap.parse_args())
#
## initialize the model to be VGG16
#Model = VGG16
#
## check to see if we are using ResNet
#if args["model"] == "resnet":
#	Model = ResNet50

# load the pre-trained CNN from disk
#model_vgg_multimodal = Caps_att(input_shape=(224,224,3), n_class=52)
model_vgg_multimodal = Caps_att(input_shape=(64,64,3),input_shape_depth=(64,64,1), n_class=108)
weights_path = 'D:/tutorial/AE-Gan_reperesentation/TFIS/kasprov_split0.7/weights-best.h5'
#for rgb
#model_vgg_multimodal.load_weights('D:/tutorial/AE-Gan_reperesentation/curtin/rgb_test1_att/weights-best.h5')
#model_vgg_multimodal.load_weights('D:/tutorial/AE-Gan_reperesentation/curtin/rgb_cam/weights-best.h5')
#model_vgg_multimodal.load_weights('D:/tutorial/AE-Gan_reperesentation/CurtinFaces_uncropped/rgb_only/weights-best.h5')
#weights_path = 'D:/tutorial/AE-Gan_reperesentation/CurtinFaces_uncropped/rgb_only/weights-best.h5'
#weights_path = 'D:/tutorial/AE-Gan_reperesentation/IIITD_rgbd/just_rgb/weights-best.h5'

#for attention
#weights_path = 'D:/tutorial/AE-Gan_reperesentation/lock3d_cropped/zhang_sav_w/o_dropout/weights-best.h5'
#weights_path = 'D:/tutorial/AE-Gan_reperesentation/CurtinFaces_cropped/attention_zhang_2_w_o_dropout/weights-best.h5'




#weights_path = 'D:/tutorial/AE-Gan_reperesentation/CurtinFaces_uncropped/vqa-att_new_dot_for_cam_1/weights-best.h5'
#model_vgg_multimodal.load_weights('D:/tutorial/AE-Gan_reperesentation/CurtinFaces_uncropped/attention_depth_cont/weights-best.h5')
#weights_path = 'D:/tutorial/AE-Gan_reperesentation/CurtinFaces_uncropped/attention_depth_cont2/weights-best.h5'
#weights_path = 'D:/tutorial/AE-Gan_reperesentation/lock3d_uncropped/attention_depth5/weights-best.h5'
#weights_path = 'D:/tutorial/AE-Gan_reperesentation/TFIS/CurtinFaces_CAM_final/weights-best.h5'

#_for rgb
#weights_path = 'D:/tutorial/AE-Gan_reperesentation/CurtinFaces_uncropped/rgb_only/weights-best.h5'





model_vgg_multimodal.load_weights(weights_path)
model_vgg_multimodal.compile(optimizer=optimizers.Adam(lr=0.0000001), loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'], metrics=['accuracy'])
model_vgg_multimodal.summary()

def read_hdf5(path):

    weights = {}

    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                print(f[key].name)
                weights[f[key].name] = f[key].value
    return weights
## load the original image from disk (in OpenCV format) and then
## resize the image to its target dimensions
#orig = cv2.imread(args["image"])
#resized = cv2.resize(orig, (224, 224))
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
# load the input image from disk (in Keras/TensorFlow format) and
    
def train_generator(batch_size, val_train): 

    dataset_dir = 'D:/RGBDI_Dataset_manual_FaceDetection_ksprov' #'D:/RGB_D_Dataset_new/fold1/train/RGB/'       #D:/CurtinFaces_crop/RGB/train/ --Curtin
    image_data_generator = data_pipeline(dataset_dir, 'kasprov', test_train='train', test_train_split=0.7)
#        print('gen init')
#        image_data_generator

    i=0
    if val_train=='train':
        while i<100:
            i = i+1
            #rgb data aug
#                print('get batch')
            x_batch_rgb, x_batch_depth, y_batch_rgb = next(image_data_generator.flow_from_dir(batch_size,(64,64,3), val_train))
#                flip_img = iaa.Fliplr(1)(images=x_batch_rgb)
#                rot_img = iaa.Affine(rotate=(-30, 30))(images=x_batch_rgb)
#
#                shear_aug = iaa.Affine(shear=(-16, 16))(images=x_batch_rgb)
#                trans_aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})(images=x_batch_rgb)
#                x_batch_rgb_final = np.concatenate([x_batch_rgb,flip_img,rot_img,shear_aug,trans_aug],axis=0)
#                y_batch_rgb_final = np.tile(y_batch_rgb,(5,1))
#                ## depth data aug
##                x_batch_depth, y_batch_depth = generator_depth.next()
#                flip_img = iaa.Fliplr(1)(images=x_batch_depth)
#                rot_img = iaa.Affine(rotate=(-30, 30))(images=x_batch_depth)
#
#                shear_aug = iaa.Affine(shear=(-16, 16))(images=x_batch_depth)
#                trans_aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})(images=x_batch_depth)
#                x_batch_depth_final = np.concatenate([x_batch_depth,flip_img,rot_img,shear_aug,trans_aug],axis=0)
#                y_batch_depth_final = np.tile(y_batch_rgb,(5,1))
#                print('batch done')
            yield [[x_batch_rgb, x_batch_depth], [y_batch_rgb,y_batch_rgb,y_batch_rgb]]
    elif val_train == 'test':
        while 1:
            x_batch_rgb, x_batch_depth, y_batch_rgb = next(image_data_generator.flow_from_dir(batch_size,(64,64,3), val_train))
#                x_batch_depth, y_batch_depth = generator_depth_val.next()
            yield [[x_batch_rgb, x_batch_depth], [y_batch_rgb,y_batch_rgb,y_batch_rgb]]


         
         
         
def save_cam(subject_no,image_no,model_vgg_multimodal,dataset,test_dir,model_type='att'):
    
    if dataset == 'CurtinFaces':
        rgb_val_dir = 'D:/CurtinFaces_crop/RGB/{}/{}/{}.jpg'.format(test_dir, subject_no, image_no)         
        depth_val_dir = 'D:/CurtinFaces_crop/DEPTH/{}/{}/{}.jpg'.format(test_dir, subject_no, image_no)  


    if dataset == 'CurtinFaces_uncropped':
        rgb_val_dir = 'D:/CurtinFaces_processed/protocol/rgb/{}/{}/{}.jpg'.format(test_dir, subject_no, image_no)
        depth_val_dir = 'D:/CurtinFaces_processed/protocol/depth/{}/{}/{}.jpg'.format(test_dir, subject_no, image_no)      
#        rgb_val_dir = 'D:/CurtinFaces_crop/RGB/{}/{}/{}.jpg'.format(test_dir, subject_no, image_no)         
#        depth_val_dir = 'D:/CurtinFaces_crop/normalized/DEPTH/{}/{}/{}.jpg'.format(test_dir, subject_no, image_no)      
#    
    
    if dataset == 'lock3d':
        rgb_val_dir = 'D:/lock3d_protocol_crop/rgb/{}/{}/{}{}_{}.jpg'.format(test_dir, subject_no, subject_no,test_dir[-2:].upper(), image_no)         
        depth_val_dir = 'D:/lock3d_protocol_crop/depth/{}/{}/{}{}_{}.jpg'.format(test_dir, subject_no, subject_no, test_dir[-2:].upper(), image_no)  
    
    if dataset == 'iiitd':
        rgb_val_dir = 'D:/RGB_D_Dataset_new/fold1/test/RGB/{}/{}.jpg'.format(subject_no, image_no)     
        depth_val_dir = 'D:/RGB_D_Dataset_new/fold1/test/depth/{}/{}.jpg'.format(subject_no, image_no)     
    #
    #
    #rgb_val_dir = 'D:/lock3d_protocol_crop/rgb/test_ps/{}/{}PS_{}.jpg'.format(subject_no,subject_no, image_no)         
    #depth_val_dir = 'D:/lock3d_protocol_crop/depth/test_ps/{}/{}PS_{}.jpg'.format(subject_no,subject_no, image_no)  
    # preprocess it
    _img = load_img(rgb_val_dir,target_size=(224,224))
    plt.imshow(_img)
    plt.show()
    img = img_to_array(_img)
    img = img.astype("float") / 255.0
    img = np.expand_dims(img, axis=0)
    #img = imagenet_utils.preprocess_input(img)
    
    
    
    input_depth_val = load_img(depth_val_dir,target_size=(224,224))
    plt.imshow(input_depth_val)
    plt.show()
    input_depth_val = img_to_array(input_depth_val)
    input_depth_val = input_depth_val.astype("float") / 255.0
    input_depth_val = np.expand_dims(input_depth_val, axis=0)
    #input_depth_val = imagenet_utils.preprocess_input(input_depth_val)
    
    #
    ## use the network to make predictions on the input imag and find
    ## the class label index with the largest corresponding probability
    if model_type == 'att':
        seed_input = [img,input_depth_val]
    else:
        seed_input = img
    preds = model_vgg_multimodal.predict(seed_input)
    i = np.argmax(preds[0])
    classIdx = i
    
    ###for rgb
    #convOutputs = get_activations(model_vgg_multimodal, img, layer_name = 'conv5_3')['conv5_3/Relu:0']
    
    
    #for attention
    if model_type == 'att':
        convOutputs = get_activations(model_vgg_multimodal, seed_input, layer_name = 'conv5_3_rgb')['conv5_3_rgb/Relu:0']
        conv_out = model_vgg_multimodal.get_layer('conv5_3_rgb').output
    else:
        convOutputs = get_activations(model_vgg_multimodal, img, layer_name = 'conv5_3')['conv5_3/Relu:0']
        conv_out = model_vgg_multimodal.get_layer('conv5_3').output
#    conv_out = model_vgg_multimodal.get_layer('conv5_3_rgb').output
    y_c = model_vgg_multimodal.output[0][0, classIdx]
    grads = K.gradients(y_c, conv_out)
    # Normalize if necessary
#    grads = normalize(grads)
    #gradient_function = K.function([model_vgg_multimodal.input], [conv_out, grads])
    
    
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    if model_type == 'att':
        evaluated_gradients = (sess.run(grads,feed_dict={model_vgg_multimodal.input[0]:seed_input[0],model_vgg_multimodal.input[1]:seed_input[1]}))
    else:
        evaluated_gradients = (sess.run(grads,feed_dict={model_vgg_multimodal.input:img}))
    sess.close()
    
    #grad_dict == convOutputs
    #grad_dict = get_gradients_of_activations(model_vgg_multimodal, seed_input, preds, layer_name='att_map')['att_map_1/mul:0']
#    evaluated_gradients = normalize(evaluated_gradients)
    grad_dict = evaluated_gradients[0]
#    reshape_shape = grad_dict.shape 
#    grad_dict = normalize(grad_dict)
#    grad_dict = np.reshape(grad_dict, reshape_shape)
    grad_dict.shape                          
    #grad_dict = get_gradients_of_trainable_weights(model_vgg_multimodal, seed_input, preds)
    if model_type == 'att':
        save_path = 'D:/TEST_FOLDER/TFIS/{}_cam_attention'.format(dataset)
    else:
        save_path = 'D:/TEST_FOLDER/TFIS/{}_cam_rgb'.format(dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    weights = np.mean(grad_dict[0], axis=(0, 1))
    cam = np.dot(convOutputs[0], weights)
    # Process CAM
#    cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR)
    cam = zoom(cam, int(224/14))
    cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
    #jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    plt.figure(figsize=(15, 10))
    plt.subplot(131)
    plt.title('GradCAM')
    plt.axis('off')
    
    plt.imshow(_img)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.savefig('{}/{}_{}_{}_gradcam'.format(save_path,test_dir,subject_no,image_no) +'.png', bbox_inches='tight')
    plt.close()
#cv2.imwrite('{}/{}_{}_gradcam.jpg'.format(save_path,subject_no,image_no), np.uint8(jetcam))
#import random
subject_no = ['01','05','10','15','20','25','30']
for subject in subject_no:
#    images_in_dir = os.listdir(os.path.join('D:/RGB_D_Dataset_new/fold1/test/RGB/',subject))
#    image_no = random.sample(images_in_dir, 5)
    image_no = ['02','04','03','13','58','68','23','33']
    for image_name in image_no:
#image_no = '0'
        save_cam(subject,image_name,model_vgg_multimodal,'CurtinFaces','test','att')
def inverse_normalization(X):
    # normalises back to ints 0-255, as more reliable than floats 0-1
    # (np.isclose is unpredictable with values very close to zero)
    result = ((X + 1.) * 127.5).astype('uint8')
    # Still check for out of bounds, just in case
    out_of_bounds_high = (result > 255)
    out_of_bounds_low = (result < 0)
    out_of_bounds = out_of_bounds_high + out_of_bounds_low
    
    if out_of_bounds_high.any():
        raise RuntimeError("Inverse normalization gave a value greater than 255")
        
    if out_of_bounds_low.any():
        raise RuntimeError("Inverse normalization gave a value lower than 1")
        
    return result        

save_dir = 'D:/TEST_FOLDER/kasprov_cam/'       
from kasprov_data_pipeline import data_pipeline
num_img=0
for  [x_batch_rgb, x_batch_depth], [y_batch_rgb,y_batch_rgb,y_batch_rgb] in  train_generator(1,'train'):      
    
    seed_input = [x_batch_rgb, x_batch_depth]
    x_batch_rgb = np.mean(x_batch_rgb, axis= 0)
    _img = inverse_normalization(x_batch_rgb)
#    plt.imshow(_img)
#    plt.show()
    preds = model_vgg_multimodal.predict(seed_input)
    i = np.argmax(preds[0])
    classIdx = i        
    convOutputs = get_activations(model_vgg_multimodal, seed_input, layer_name = 'conv5_3_rgb')['conv5_3_rgb/Relu:0']
    conv_out = model_vgg_multimodal.get_layer('conv5_3_rgb').output
    y_c = model_vgg_multimodal.output[0][0, classIdx]
    grads = K.gradients(y_c, conv_out)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    evaluated_gradients = (sess.run(grads,feed_dict={model_vgg_multimodal.input[0]:seed_input[0],model_vgg_multimodal.input[1]:seed_input[1]}))
    sess.close()
    grad_dict = evaluated_gradients[0]
    grad_dict.shape
    weights = np.mean(grad_dict[0], axis=(0, 1))
    cam = np.dot(convOutputs[0], weights)
    cam = zoom(cam, int(224/14))
    cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
    #jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    plt.figure(figsize=(15, 10))
    plt.subplot(131)
    plt.title('GradCAM')
    plt.axis('off')
    
    plt.imshow(_img)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.savefig('{}/{}_{}_gradcam'.format(save_dir,classIdx,num_img) +'.png', bbox_inches='tight')
    plt.close()
    num_img= num_img +1