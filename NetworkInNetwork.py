# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:28:54 2020

@author: Tanmay Thakur
"""

import tensorflow as tf
import numpy as np


def preprocess():
        (x_train,y_train),(x_test,y_test)=tf.contrib.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32')/255
        x_test = x_test.astype('float32')/255
        x_train = np.reshape(x_train,(len(x_train),28,28,1))
        x_test = np.reshape(x_test,(len(x_test),28,28,1))
        y_train = tf.contrib.keras.utils.to_categorical(y_train,10)
        y_test = tf.contrib.keras.utils.to_categorical(y_test,10)
        return (x_train,y_train,x_test,y_test)


def model(x):
        x1=tf.contrib.keras.layers.Conv2D(11,(1,1),padding='same')(x)
        x1=tf.contrib.keras.layers.Flatten()(x1)
        x1=tf.contrib.keras.layers.Dense(900,activation='relu')(x1)
        x1=tf.contrib.keras.layers.Dense(900,activation='relu')(x1)
        x1=tf.contrib.keras.layers.Reshape((30,30,1),input_shape=x1.shape)(x1)
        x1=tf.contrib.keras.layers.Flatten()(x1)
        x1=tf.contrib.keras.layers.Dense(400,activation='relu')(x1)
        x1=tf.contrib.keras.layers.Dense(400,activation='relu')(x1)
        x1=tf.contrib.keras.layers.Reshape((20,20,1),input_shape=x1.shape)(x1)
        x1=tf.contrib.keras.layers.Conv2D(11,(1,1),padding='same')(x1)
        x1=tf.contrib.keras.layers.Flatten()(x1)
        x1=tf.contrib.keras.layers.Dense(100,activation='relu')(x1)
        x1=tf.contrib.keras.layers.Dense(100,activation='relu')(x1)
        x1=tf.contrib.keras.layers.Reshape((10,10,1),input_shape=x1.shape)(x1)
        x1=tf.contrib.keras.layers.Conv2D(11,(1,1),padding='same')(x1)
        x1=tf.contrib.keras.layers.AveragePooling2D((2,2),padding='same')(x1)
        x1=tf.contrib.keras.layers.Flatten()(x1)
        x1=tf.contrib.keras.layers.Dense(10,activation='relu')(x1)
        output=tf.contrib.keras.layers.Activation('softmax')(x1)
        return output

input_img = tf.contrib.keras.layers.Input(shape=(28,28,1))
NiN = tf.contrib.keras.models.Model(input_img,model(input_img))
NiN.compile(optimizer='adadelta',loss='categorical_crossentropy')
x_train,y_train,x_test,y_test = preprocess()
NiN.fit(x_train,y_train,epochs=10,batch_size=128,shuffle=True)