# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 01:56:09 2019

@author: hardi
"""
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Reshape, Dense, LSTM, Bidirectional, multiply, Permute, Concatenate, Conv2D, Conv1D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
from keras import models 
from keras.layers import concatenate , BatchNormalization, dot

def cbam_block(cbam_feature, ratio=8, att_type='Both'):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	if att_type == 'Channel':
		cbam_feature = channel_attention(cbam_feature, ratio)
	elif att_type == 'Spatial':
		cbam_feature = spatial_attention(cbam_feature)
	elif att_type == 'Both':
		cbam_feature = channel_attention(cbam_feature, ratio)
		cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature, ratio=8):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def channel_attention_lstm(input_feature):
#	input_feature = Input(shape=(7,7,1024))
	channel_axis = -1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]
	spatial = input_feature._keras_shape[1]
	att_layer = Dense(channel,
							 activation='tanh',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')    
	reshape_feature = Reshape(target_shape=[spatial*spatial,channel], name='reshape_channel')(input_feature)
#    
	lstm_feature = LSTM(channel, activation='tanh', input_shape=(spatial*spatial,1))(reshape_feature)
#	lstm_feature1 = LSTM(channel, activation='tanh', return_sequences=True)(lstm_feature)
#	lstm_feature2 = LSTM(channel, activation='tanh')(lstm_feature1)    
	att_feature = att_layer(lstm_feature)
	att_weights = Activation('sigmoid')(att_feature)
#	train_model = models.Model(inputs=[input_feature], outputs=[lstm_feature])
#	train_model.summary()
	return multiply([input_feature, att_weights])


def spatial_attention_fc(input_feature):
#	kernel_size = 7
	

#	channel = input_feature._keras_shape[-1]
	spatial = input_feature._keras_shape[1]

	
#	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input_feature)
#	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input_feature)
#	assert max_pool._keras_shape[-1] == 1
#	concat = Concatenate(axis=3)([avg_pool, max_pool])
#	assert concat._keras_shape[-1] == 2
	reshape_feature = Reshape(target_shape=[spatial*spatial], name='reshape_spatial')(max_pool)

	fc_att = Dense(spatial*spatial, activation='softmax', name='fc_att')(reshape_feature)
	reshape_att = Reshape(target_shape=[spatial,spatial,1], name='reshape_att')(fc_att)
#	cbam_feature = Conv2D(filters = 1,
#					kernel_size=kernel_size,
#					strides=1,
#					padding='same',
#					activation='sigmoid',
#					kernel_initializer='he_normal',
#					use_bias=False)(concat)	
#	assert cbam_feature._keras_shape[-1] == 1
	

	return multiply([input_feature, reshape_att])


def spatial_attention(input_feature):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature._keras_shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature._keras_shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature._keras_shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])

def spatial_attention_weights(input_feature):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature._keras_shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature._keras_shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature._keras_shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return cbam_feature


def cross_attention(input_img_feature,input_ref_feature, ratio=[1,1]):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_img_feature._keras_shape[channel_axis]
	spatial = input_img_feature._keras_shape[1]
#	batch = input_img_feature._keras_shape[0]
#    debugging
#	input_img_feature = Input(shape=(7,7,512))
#	input_ref_feature = Input(shape=(7,7,512))
    
	layer_img = Dense(channel// ratio[0],
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	layer_ref = Dense(channel// ratio[0],
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel // ratio[1],
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')

	
#	avg_pool = GlobalAveragePooling2D()(input_feature)    
#	avg_pool = Reshape((1,1,channel))(avg_pool)
#	assert avg_pool._keras_shape[1:] == (1,1,channel)
	layer_img_mat = layer_img(input_img_feature)
	layer_img_act = Activation('tanh')(layer_img_mat)

	layer_ref_mat = layer_ref(input_ref_feature)
	layer_ref_act = Activation('tanh')(layer_ref_mat)

    
	reshape_feature_img = Reshape(target_shape=[ spatial*spatial ,channel// ratio[0]], name='reshape1')(layer_img_act)
	reshape_feature_ref = Reshape(target_shape=[ spatial*spatial , channel// ratio[0]], name='reshape2')(layer_ref_act)
    
#	mul_layers = multiply([layer_img_act, layer_ref_act]) --- elementwise
	mul_layers = dot([reshape_feature_img, reshape_feature_ref],axes=2)  ###---dot product similarity measure

	layer2 = shared_layer_two(mul_layers)
	
	reshape_feature_layer2 = Reshape(target_shape=[ spatial , spatial , channel// ratio[1]], name='reshape3')(layer2)
	conv_feature = Conv2D(filters = 1,
					kernel_size=1,
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(reshape_feature_layer2)	
	bn_conv = BatchNormalization(axis=3)(conv_feature)
	act_conv = Activation('softmax')(bn_conv)
	rgb_att = multiply([input_img_feature, act_conv])
	depth_att = multiply([input_ref_feature, act_conv])
    
    ##debugging part
#	train_model = models.Model(inputs=[input_img_feature, input_ref_feature], outputs=[act_conv])
#	train_model.summary()
    
	return concatenate([rgb_att,depth_att], axis=-1)



def mlb_attention(input_img_feature,input_ref_feature, ratio=[1,1]):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_img_feature._keras_shape[channel_axis]
	spatial = input_img_feature._keras_shape[1]
#	batch = input_img_feature._keras_shape[0]
#    debugging
#	input_img_feature = Input(shape=(7,7,512))
#	input_ref_feature = Input(shape=(7,7,512))
    
	layer_img = Dense(channel// ratio[0],
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	layer_ref = Dense(channel// ratio[0],
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel // ratio[1],
                             activation='softmax',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')

	
#	avg_pool = GlobalAveragePooling2D()(input_feature)    
#	avg_pool = Reshape((1,1,channel))(avg_pool)
#	assert avg_pool._keras_shape[1:] == (1,1,channel)
	layer_img_mat = layer_img(input_img_feature)
#	layer_img_act = Activation('tanh')(layer_img_mat)

	layer_ref_mat = layer_ref(input_ref_feature)
#	layer_ref_act = Activation('tanh')(layer_ref_mat)

    
#    layer2 = shared_layer_two(mul_layers)
    
    
#    
	reshape_feature_img = Reshape(target_shape=[ spatial*spatial ,channel// ratio[0]], name='reshape1')(layer_img_mat)
	reshape_feature_ref = Reshape(target_shape=[ spatial*spatial , channel// ratio[0]], name='reshape2')(layer_ref_mat)
    
#	mul_layers = multiply([layer_img_act, layer_ref_act]) #--- elementwise
	mul_layers = dot([reshape_feature_img, reshape_feature_ref],axes=2)  ###---dot product similarity measure
#
	layer2 = shared_layer_two(mul_layers)
#	
	reshape_feature_layer2 = Reshape(target_shape=[ spatial , spatial , channel// ratio[1]], name='reshape3')(layer2)
	act_conv = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(reshape_feature_layer2)
#	conv_feature = Conv2D(filters = 1,
#					kernel_size=1,
#					strides=1,
#					padding='same',
#					kernel_initializer='he_normal',
#					use_bias=False)(reshape_feature_layer2)	
#	bn_conv = BatchNormalization(axis=3)(conv_feature)
#	act_conv = Activation('softmax')(bn_conv)

    
    ##debugging part
#	train_model = models.Model(inputs=[input_img_feature, input_ref_feature], outputs=[act_conv])
#	train_model.summary()
    
	return multiply([input_img_feature, act_conv], name='att_map')	

def senet_attention(input_img_feature,input_ref_feature, ratio=[1,1]):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_img_feature._keras_shape[channel_axis]
	spatial = input_img_feature._keras_shape[1]
#	batch = input_img_feature._keras_shape[0]
#    debugging
#	input_img_feature = Input(shape=(7,7,512))
#	input_ref_feature = Input(shape=(7,7,512))
    
	layer_img = Dense(channel// ratio[0],
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	layer_ref = Dense(channel// ratio[0],
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel // ratio[1],
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')

	
#	avg_pool = GlobalAveragePooling2D()(input_feature)    
#	avg_pool = Reshape((1,1,channel))(avg_pool)
#	assert avg_pool._keras_shape[1:] == (1,1,channel)
	layer_img_mat = layer_img(input_img_feature)
#	layer_img_act = Activation('tanh')(layer_img_mat)

	layer_ref_mat = layer_ref(input_ref_feature)
#	layer_ref_act = Activation('tanh')(layer_ref_mat)

    
#    layer2 = shared_layer_two(mul_layers)
    
    
#    
	reshape_feature_img = Reshape(target_shape=[ spatial*spatial ,channel// ratio[0]], name='reshape1')(layer_img_mat)
	reshape_feature_ref = Reshape(target_shape=[ spatial*spatial , channel// ratio[0]], name='reshape2')(layer_ref_mat)
    
#	mul_layers = multiply([layer_img_act, layer_ref_act]) #--- elementwise
	mul_layers = dot([reshape_feature_img, reshape_feature_ref],axes=2)  ###---dot product similarity measure
#
	layer2 = shared_layer_two(mul_layers)
#	
	reshape_feature_layer2 = Reshape(target_shape=[ spatial , spatial , channel// ratio[1]], name='reshape3')(layer2)
	conv_feature = Conv2D(filters = 1,
					kernel_size=1,
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(reshape_feature_layer2)	
	bn_conv = BatchNormalization(axis=3)(conv_feature)
	act_conv = Activation('softmax')(bn_conv)

    
    ##debugging part
#	train_model = models.Model(inputs=[input_img_feature, input_ref_feature], outputs=[act_conv])
#	train_model.summary()
    
	return multiply([input_img_feature, act_conv], name='att_map')	


def lstm_attention(input_img_feature,input_ref_feature, ratio=[1,1]):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_img_feature._keras_shape[channel_axis]
	spatial = input_img_feature._keras_shape[1]
#	batch = input_img_feature._keras_shape[0]
#    debugging
#	input_img_feature = Input(shape=(7,7,512))
#	input_ref_feature = Input(shape=(7,7,512))
    
	layer_img = Dense(channel// ratio[0],
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	layer_ref = Dense(channel// ratio[0],
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel // ratio[1],
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	layer_three = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')    

	
#	avg_pool = GlobalAveragePooling2D()(input_feature)    
#	avg_pool = Reshape((1,1,channel))(avg_pool)
#	assert avg_pool._keras_shape[1:] == (1,1,channel)
	layer_img_mat = layer_img(input_img_feature)
	layer_img_act = Activation('tanh')(layer_img_mat)

	layer_ref_mat = layer_ref(input_ref_feature)
	layer_ref_act = Activation('tanh')(layer_ref_mat)

    
#    layer2 = shared_layer_two(mul_layers)
    
    
#    
	reshape_feature_img = Reshape(target_shape=[ spatial*spatial ,channel// ratio[0]], name='reshape1')(layer_img_act)
	reshape_feature_ref = Reshape(target_shape=[ spatial*spatial , channel// ratio[0]], name='reshape2')(layer_ref_act)
    
#	mul_layers = multiply([layer_img_act, layer_ref_act]) #--- elementwise
	mul_layers = dot([reshape_feature_img, reshape_feature_ref],axes=2)  ###---dot product similarity measure
#
	layer2 = shared_layer_two(mul_layers)
#	
	reshape_feature_layer2 = Reshape(target_shape=[ spatial , spatial , channel// ratio[1]], name='reshape3')(layer2)
	conv_feature = Conv2D(filters = channel,
					kernel_size=1,
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(reshape_feature_layer2)	
	bn_conv = BatchNormalization(axis=3)(conv_feature)
	act_conv = Activation('softmax')(bn_conv)

	reshape_feature = Reshape(target_shape=[channel, spatial*spatial], name='reshape')(act_conv)
#    
	lstm_feature = Bidirectional(LSTM(channel, activation='relu'), input_shape=(spatial*spatial,1))(reshape_feature)
	layer2 = layer_three(lstm_feature)
#	reshape_feature2 = Reshape(target_shape=[-1, channel, spatial, spatial], name='reshape2')(lstm_feature)
##	layer2 = shared_layer_three(reshape_feature2)
##	cbam_feature = Add()([avg_pool,max_pool])
	lstm_att_weights = Activation('softmax')(layer2)
	att_feature = multiply([conv_feature, lstm_att_weights])	
	att_mean_map = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(att_feature)
    ##debugging part
#	train_model = models.Model(inputs=[input_img_feature, input_ref_feature], outputs=[att_mean_map])
#	train_model.summary()
    
	return multiply([input_img_feature, att_feature])		
#	