#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D,     AveragePooling2D, Flatten, Activation

import cv2
import numpy as np
from keras.utils import np_utils

import math
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import matplotlib

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from PIL import Image

from keras.layers.normalization import BatchNormalization


# In[2]:


# Read Train set - folder
train_folder_path = './data/PE92_train_598_aug/''
categories = []

# If file changes, num_classes changes
with open('./data/files_598.txt', 'r') as f:
    infoFile = f.readlines()
    
    for line in infoFile:
        words = line.split()
        categories.append(words[0])

num_classes = len(categories)


# In[3]:


# Read Train set - Data
# Color Scale로 읽음 - 모델에 사용
# 논문에서의 image size: 60x60
image_w = 60
image_h = 60

X = []
Y = []

for idex, category in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = train_folder_path + category + '/'
  
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            image_path = image_dir+filename

            n = np.fromfile(image_path, dtype=np.uint8)
            
            # COLOR SCALE로 READ (255: white, 0: black)
            img = cv2.imdecode(n, flags=cv2.IMREAD_COLOR)
            
            # 논문 사이즈에 맞도록 resizing - 선형보간법 적용
            img = cv2.resize(img, None, fx=image_w/img.shape[0], fy=image_h/img.shape[1], interpolation=cv2.INTER_AREA)          
            
            X.append(img/255)
            Y.append(label)
            
X = np.array(X)
Y = np.array(Y)


# In[4]:


# train set:validation set = 8:2 => 9:1
X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.1)


# In[5]:


# Read Test set - folder 
test_folder_path = './data/PE92_test_598'
categories = []

# If file changes, num_classes changes
with open('./data/files_598.txt', 'r') as f:
    infoFile = f.readlines()
    
    for line in infoFile:
        words = line.split()
        categories.append(words[0])

num_classes = len(categories)


# In[6]:


# Read Test set - Data

image_w = 60
image_h = 60

X_test = []
Y_test = []

for idex, category in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = test_folder_path + category + '/'
  
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            image_path = image_dir+filename
            
            n = np.fromfile(image_path, dtype=np.uint8)
            
            # COLOR SCALE로 READ / 255: white, 0: black
            img = cv2.imdecode(n, flags=cv2.IMREAD_COLOR)
            
            # 논문에 맞게 image resizing
            img = cv2.resize(img, None, fx=image_w/img.shape[0], fy=image_h/img.shape[1], interpolation=cv2.INTER_AREA)
           
            X_test.append(img/255)
            Y_test.append(label)
            
X_test = np.array(X_test)
Y_test = np.array(Y_test)


# In[7]:


# define inception v1 architecture 

# 5x5 filter를 3x3 - 3x3로 수정
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None,
                     kernel_init='glorot_uniform', bias_init='zeros'):

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3_reduce)
    
    # 위의 conv 3x3 결과에 filters_5x5_reduce 값으로 1x1 conv - filters_5x5 값으로 3x3 conv 한 번 더 
    conv_3x3_reduce_2 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)
    conv_3x3_2 = Conv2D(filters_5x5, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3_reduce_2)
    
    max_pool = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)

    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(max_pool)

    output = concatenate([conv_1x1, conv_3x3, conv_3x3_2, pool_proj], axis=3, name=name)

    return output


# In[8]:


# 논문의 학습률 적용 - cliff drop 방식
def decay(epoch, steps=100):
    
    # 논문의 학습률 감소 적용 0.01 -> 0.001
    initial_lrate = 0.001
    
    # 논문의 cliff drop 방법 적용
    if epoch < 31:
        Irate = 0.001
        
    elif (epoch >= 31 and epoch < 61):
        Irate = 0.0005
        
    else:
        Irate = 0.00001
    
    return Irate


# In[9]:


def main():

    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)
    
    # input shape: 60x60 (channel 3)
    input_layer = Input(shape=(image_w, image_h, 3))

    # Conv 7x7 ~ maxpool 3x3/2 => Conv 5x5 - maxpool 3x3/2 변형 적용
    # Layer 1 - batch normalization 적용 [Conv-BatchNormalization-Activation-MaxPool2D]
    x = Conv2D(64,(5,5),strides=(1,1),name='conv_1_5x5/1')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), name='max_pool_1_3x3/2', padding='same')(x)
    
    # 보조 분류기를 배치하지 않았다고 함 => auxiliary classifier 제거 
    # Layer 2
    x = inception_module(x, 64, 96, 128, 16, 32, 32, name='inception_3a', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 128, 128, 192, 32, 96, 64, name='inception_3b', kernel_init=kernel_init, bias_init=bias_init)
    x = MaxPool2D((3, 3), strides=(2, 2), name='max_pool_3_3x3/2')(x)

    # Layer 3
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_4a')
    x = inception_module(x, 160, 112, 224, 24, 64, 64, name='inception_4b', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 128, 128, 256, 24, 64, 64, name='inception_4c', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 112, 144, 288, 32, 64, 64, name='inception_4d', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_4e', kernel_init=kernel_init, bias_init=bias_init)
    x = MaxPool2D((3, 3), strides=(2, 2), name='max_pool_4_3x3/2')(x)

    # Layer 4
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_5a', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 384, 192, 384, 48, 128, 128, name='inception_5b', kernel_init=kernel_init, bias_init=bias_init)
    # AveragePooling2D((7,7), strides=(1,1)) 이 적용되었지만 Negative Dimension 오류 발생 -> 전 단계의 Output Shape인 (6x6)으로 변경
    x = AveragePooling2D((6,6), strides=(1,1), name='avg_pool_6x6/1')(x)
    # AverPool 이후 GlobalAveragePooling 적용
    x = GlobalAveragePooling2D(name='global_avg_pool_5_3x3/1')(x)
    
    x = Dropout(0.4)(x)
    output_tensor = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(input_layer, output_tensor, name='Final_598_aug')

    model.summary()

    epoch = 100
    initial_lrate = 0.001

    sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)
    lr_sc = LearningRateScheduler(decay, verbose=1)

    model.compile(loss='categorical_crossentropy',loss_weights=[1], optimizer=sgd, metrics=['accuracy'])
    
    hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                       epochs=epoch, batch_size=128, callbacks=[lr_sc])
    
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print("test_acc:", test_acc)
    
    model.save('Final_598_aug.h5')


# In[10]:


if __name__ == '__main__':
    main()

