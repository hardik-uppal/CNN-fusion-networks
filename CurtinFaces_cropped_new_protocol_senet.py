# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:10:40 2019

@author: hardi
"""

import gc
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
#from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
#from keras_vggface.vggface import VGGFace
from model_vgg_face import VGG16
from model_resnet import RESNET50
from model_senet import SENET50
import keras
import pickle
from attention_module import cbam_block, spatial_attention, channel_attention, spatial_attention_weights, mlb_attention,lstm_attention
#import config
from keras.models import Model
from keras.layers import multiply, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation, Dropout, BatchNormalization
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
from keras_vggface import utils
from keras.utils.data_utils import get_file
import imgaug.augmenters as iaa

K.set_image_data_format('channels_last')

def reset_keras(model):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
    
    try:
        del model # this is from global space - change this as you need
    except:
        pass
    
    print(gc.collect()) # if it's done something you should see a number being outputted
    
    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))    
    
    

def deepnet(input_shape, n_class):
    """
    
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    
    :return:  Keras Model used for training
    """
    # RGB MODALITY BRANCH OF CNN
    inputs_rgb = layers.Input(shape=input_shape)
    ########################VGG/RESNET or any other network
    vgg_model_rgb = RESNET50(include_top=False, weights='vggface', input_tensor=None, input_shape=input_shape, pooling=None, type_name='rgb')
    conv_model_rgb = vgg_model_rgb(inputs_rgb)
#    attention_spatial_rgb = spatial_attention(conv_model_rgb)
    
    ########################Depth MODALITY BRANCH OF CNN
    
#    inputs_depth = layers.Input(shape=input_shape)
#    vgg_model_depth = VGG16(include_top=False, weights='vggface', input_tensor=None, input_shape=input_shape, pooling=None, type_name='depth')
#    conv_model_depth = vgg_model_depth(inputs_depth)

    
    
#    attention_spatial_depth = spatial_attention_weights(conv_model_depth)


    # CONACTENATE the ends of RGB & DEPTH 

#    merge_rgb_depth = layers.concatenate([conv_model_rgb,conv_model_depth], axis=-1)
#    merge_rgb_depth = layers.concatenate([attention_spatial_rgb,attention_spatial_depth], axis=-1)
 
    
    
    ### Attention mechanism
    
    
#    primarycaps = PrimaryCap(merge_rgb_depth, dim_capsule=16, n_channels=32, kernel_size=3, strides=1, padding='valid')
#    secondarycaps = PrimaryCap(primarycaps, dim_capsule=8, n_channels=32, kernel_size=3, strides=1, padding='valid')
#    idcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=32, routings=3, name='idcaps')(primarycaps)
#    attention_features = channel_attention(conv_model_rgb)
#    attention_features = cbam_block(conv_model_rgb)
    ## new attention mech
#    new_att_features= multiply([conv_model_rgb, conv_model_depth])
#  
#    attention_features = cbam_block(merge_rgb_depth)
    
#    
#    attention_features = mlb_attention(conv_model_rgb,conv_model_depth, ratio=[2,1])
    
    
#    attention_features_cross = cross_attention(conv_model_depth,conv_model_rgb, ratio=[2,1])
#    primarycaps = PrimaryCap(attention_features, dim_capsule=8, n_channels=32, kernel_size=3, strides=1, padding='valid')
#    idcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=3, name='idcaps')(primarycaps)
    
#    merge_rgb_depth = layers.concatenate([attention_features_rgb,attention_features_depth], axis=-1)
    ######## Common network
    flat_model = layers.Flatten(name='flatten')(conv_model_rgb)
    
    
    
    # removed for capsule
#    fc6 = layers.Dense(2048, activation='relu', name='fc6')(flat_model)
#    bn_1 = BatchNormalization(name='1_bn')(fc6)
#    dropout_1 = layers.Dropout(0.5)(bn_1)
#    
#    
#    
#
#
#    fc7 = layers.Dense(1024, activation='relu', name='fc7')(dropout_1)
#    bn_2 = BatchNormalization(name='2_bn')(fc7)
#    dropout_2 = layers.Dropout(0.5)(bn_2)
#    
#    fc8 = layers.Dense(1024, activation='relu', name='fc8')(dropout_2)
#    bn_3 = BatchNormalization(name='3_bn')(fc8)
#    dropout_3 = layers.Dropout(0.5)(bn_3)
###    

    
    #VECTORIZING OUTPUT
    output = layers.Dense(n_class, activation='softmax', name='output')(flat_model)
    
    # MODAL [INPUTS , OUTPUTS]
    train_model = models.Model(inputs=[inputs_rgb], outputs=[output])
    
#    weights_path = 'CurtinFaces/vgg_multimodal_dropout-0.5_3fc-512/weights-25.h5'
#    train_model.load_weights(weights_path)
    train_model.summary()
#    for layer in train_model.layers[:-2]:
#        layer.trainable = False
#    for layer in train_model.layers[11]:
#    train_model.layers[11].trainable = False
##    for layer in train_model.layers[14]:
#    train_model.layers[11].trainable = False
#    for layer in train_model.layers[1].layers[-10:]:
#        layer.trainable = True
#    for layer in train_model.layers[3].layers[:-4]:
#        layer.trainable = False




    return train_model

#def margin_loss(y_true, y_pred):
#    """
#    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
#    :param y_true: [None, n_classes]
#    :param y_pred: [None, num_capsule]
#    :return: a scalar loss value.
#    """
#    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
#        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
#
#    return K.mean(K.sum(L, 1))


def train(model, args):
    """
    Training 
    :param model: the  model
    
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
#    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    es_cb = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-best.h5', monitor='val_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss={'output':'categorical_crossentropy'},#triplet_loss_adapted_from_tf,
#                  loss_weights={'output':0.5, 'output_rgb': 0.25,'output_depth':0.25},
                  metrics=['accuracy'])

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(batch_size, val_train): 

        rgb_train_dir = 'D:/CurtinFaces_crop_1/RGB/train_new/' #'D:/RGB_D_Dataset_new/fold1/train/RGB/'       #D:/CurtinFaces_crop/RGB/train/ --Curtin
        depth_train_dir = 'D:/CurtinFaces_crop_1/DEPTH/train_new/'   #D:/CurtinFaces_crop/normalized/DEPTH/train/
        rgb_val_dir = 'D:/CurtinFaces_crop_1/RGB/test_ex/'          #D:/CurtinFaces_crop/RGB/test1/
        depth_val_dir = 'D:/CurtinFaces_crop_1/DEPTH/test_ex/'     #D:/CurtinFaces_crop/normalized/DEPTH/test1/
        batch_size = int(batch_size/5)
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)#, validation_split=0.2)  
        generator_rgb = train_datagen.flow_from_directory(directory= rgb_train_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory= depth_train_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        
        generator_rgb_val = train_datagen.flow_from_directory(directory= rgb_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        generator_depth_val = train_datagen.flow_from_directory(directory= depth_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42)
        if val_train=='train':
            while 1:
                #rgb data aug
                x_batch_rgb, y_batch_rgb = generator_rgb.next()
                flip_img = iaa.Fliplr(1)(images=x_batch_rgb)
                rot_img = iaa.Affine(rotate=(-30, 30))(images=x_batch_rgb)

                shear_aug = iaa.Affine(shear=(-16, 16))(images=x_batch_rgb)
                trans_aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})(images=x_batch_rgb)
                x_batch_rgb_final = np.concatenate([x_batch_rgb,flip_img,rot_img,shear_aug,trans_aug],axis=0)
                y_batch_rgb_final = np.tile(y_batch_rgb,(5,1))
                ## depth data aug
                x_batch_depth, y_batch_depth = generator_depth.next()
                flip_img = iaa.Fliplr(1)(images=x_batch_depth)
                rot_img = iaa.Affine(rotate=(-30, 30))(images=x_batch_depth)

                shear_aug = iaa.Affine(shear=(-16, 16))(images=x_batch_depth)
                trans_aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})(images=x_batch_depth)
                x_batch_depth_final = np.concatenate([x_batch_depth,flip_img,rot_img,shear_aug,trans_aug],axis=0)
                y_batch_depth_final = np.tile(y_batch_rgb,(5,1))
                yield [x_batch_rgb_final, y_batch_rgb_final]
        elif val_train == 'val':
            while 1:
                x_batch_rgb, y_batch_rgb = generator_rgb_val.next()
                x_batch_depth, y_batch_depth = generator_depth_val.next()
                yield [x_batch_rgb, y_batch_rgb] 
            
    
            
    # Training with data augmentation. 
    model.fit_generator(generator=train_generator(args.batch_size,'train'),
                        steps_per_epoch=int(208 / int(args.batch_size/5)),##936 curtin faces###424 fold1 iiitd ##46846
                        epochs=args.epochs,
                        validation_data=train_generator(args.batch_size,'val'),
                        validation_steps = int(312 / int(args.batch_size)),##4108 curtin faces###4181 fold1 iiitd ##39942
                        callbacks=[log, tb, checkpoint, lr_decay, es_cb])
    # End: Training with data augmentation -----------------------------------------------------------------------#

#    model.save_weights(args.save_dir + '/trained_model.h5')
#    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, args):

#    
#    from incorrect_pairs_save import get_pairs
#    
#    file_path = 'D:/TEST_FOLDER/curtin/Incorrect_examples'
#    
#    if not os.path.exists(file_path):
#        os.makedirs(file_path)
        
        
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
              loss=['categorical_crossentropy'],
              
              metrics=['accuracy'])
#    model.load_weights('./CurtinFaces_uncropped/vqa-att_new_dot/weights-best.h5')
    
### after defining generator    
#    get_pairs(test_generator_2, model, file_path, 'iiit2')    
#    
    def test_generator_il(batch_size=1):
        rgb_val_dir = 'D:/CurtinFaces_crop_1/RGB/test_il/'          #D:/CurtinFaces_crop/RGB/test1/
        depth_val_dir = 'D:/CurtinFaces_crop_1/DEPTH/test_il/'     #D:/CurtinFaces_crop/normalized/DEPTH/test1/

        train_datagen = ImageDataGenerator(rescale=1./255)  
        generator_rgb = train_datagen.flow_from_directory(directory=rgb_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory=depth_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        i =0
        while i<260:
            i =i+1
#        while 1:
            x_batch_rgb, y_batch_rgb = generator_rgb.next()
            x_batch_depth, y_batch_depth = generator_depth.next()
            yield [x_batch_rgb, y_batch_rgb] 
    
    def test_generator_il_ex(batch_size=1):
        rgb_val_dir = 'D:/CurtinFaces_crop_1/RGB/test_il_ex/'          #D:/CurtinFaces_crop/RGB/test1/
        depth_val_dir = 'D:/CurtinFaces_crop_1/DEPTH/test_il_ex/'     #D:/CurtinFaces_crop/normalized/DEPTH/test1/

        train_datagen = ImageDataGenerator(rescale=1./255)  
        generator_rgb = train_datagen.flow_from_directory(directory=rgb_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory=depth_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        i =0
        while i<1560:
            i =i+1
#        while 1:
            x_batch_rgb, y_batch_rgb = generator_rgb.next()
            x_batch_depth, y_batch_depth = generator_depth.next()
            yield [x_batch_rgb, y_batch_rgb] 

    def test_generator_ps_ex(batch_size=1):
        rgb_val_dir = 'D:/CurtinFaces_crop_1/RGB/test_ps_ex/'          #D:/CurtinFaces_crop/RGB/test1/
        depth_val_dir = 'D:/CurtinFaces_crop_1/DEPTH/test_ps_ex/'     #D:/CurtinFaces_crop/normalized/DEPTH/test1/

        train_datagen = ImageDataGenerator(rescale=1./255)  
        generator_rgb = train_datagen.flow_from_directory(directory=rgb_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory=depth_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        i =0
        while i<1872:
            i =i+1
#        while 1:
            x_batch_rgb, y_batch_rgb = generator_rgb.next()
            x_batch_depth, y_batch_depth = generator_depth.next()
            yield [x_batch_rgb, y_batch_rgb] 

    def test_generator_ps(batch_size=1):
        rgb_val_dir = 'D:/CurtinFaces_crop_1/RGB/test_ps_new/'          #D:/CurtinFaces_crop/RGB/test1/
        depth_val_dir = 'D:/CurtinFaces_crop_1/DEPTH/test_ps_new/'     #D:/CurtinFaces_crop/normalized/DEPTH/test1/

        train_datagen = ImageDataGenerator(rescale=1./255)  
        generator_rgb = train_datagen.flow_from_directory(directory=rgb_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory=depth_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        i =0
        while i<416:
            i =i+1
#        while 1:
            x_batch_rgb, y_batch_rgb = generator_rgb.next()
            x_batch_depth, y_batch_depth = generator_depth.next()
            yield [x_batch_rgb, y_batch_rgb]            
    def test_generator_ex(batch_size=1):
        rgb_val_dir = 'D:/CurtinFaces_crop_1/RGB/test_ex/'          #D:/CurtinFaces_crop/RGB/test1/
        depth_val_dir = 'D:/CurtinFaces_crop_1/DEPTH/test_ex/'     #D:/CurtinFaces_crop/normalized/DEPTH/test1/

        train_datagen = ImageDataGenerator(rescale=1./255)  
        generator_rgb = train_datagen.flow_from_directory(directory=rgb_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        generator_depth = train_datagen.flow_from_directory(directory=depth_val_dir, target_size=(224, 224), color_mode="rgb",
                                                      batch_size=1, class_mode="categorical", shuffle=True, seed=42)
        i =0
        while i<312:
            i =i+1
#        while 1:
            x_batch_rgb, y_batch_rgb = generator_rgb.next()
            x_batch_depth, y_batch_depth = generator_depth.next()
            yield [x_batch_rgb, y_batch_rgb]             
    scores = model.evaluate_generator(generator=test_generator_il(1),steps = 260)###test1 2028 ###test2 1560##test3 260
    print('Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1:]))
    import csv
    test_log = args.save_dir + '/log_test_il.csv'
    with open(test_log, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1:])])
        
    scores = model.evaluate_generator(generator=test_generator_il_ex(1),steps = 1560)###test1 2028 ###test2 1560##test3 260
    print('Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1:]))
    import csv
    test_log = args.save_dir + '/log_test_il_ex.csv'
    with open(test_log, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1:])])
    scores = model.evaluate_generator(generator=test_generator_ps_ex(1),steps = 1872)###test1 2028 ###test2 1560##test3 260
    print('Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1:]))
    import csv
    test_log = args.save_dir + '/log_test_ps_ex.csv'
    with open(test_log, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1:])])
    scores = model.evaluate_generator(generator=test_generator_ps(1),steps = 416)###test1 2028 ###test2 1560##test3 260
    print('Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1:]))
    import csv
    test_log = args.save_dir + '/log_test_ps.csv'
    with open(test_log, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1:])])
    scores = model.evaluate_generator(generator=test_generator_ex(1),steps = 312)###test1 2028 ###test2 1560##test3 260
    print('Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1:]))
    import csv
    test_log = args.save_dir + '/log_test_ex.csv'
    with open(test_log, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['Test loss: {} ; Accuracy on Test: {}'.format(scores[0],scores[1:])])

if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="RGB-D network")
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=10, type=int)## only divisible by 5
    parser.add_argument('--lr', default=0.00001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./CurtinFaces_cropped_new/resnet_rgb_train_new')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)



    # define model
    model = deepnet(input_shape=(224,224,3), n_class=52)
    model.summary()
#    model.load_weights('./CurtinFaces_cropped_new/senet_rgb_train_new/weights-best.h5')


    model_trained = train(model=model, args=args)
    
#    model_trained = VGGFace_multimodal(input_shape=(224,224,3), n_class=52)
    test(model=model, args=args)
    reset_keras(model=model)

#    reset_keras(model=model)
  # as long as weights are given, will run testing
