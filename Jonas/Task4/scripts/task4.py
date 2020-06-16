#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "1.0.1"
__maintainer__ = "Jonas Lussi"
__email__ = "jlussi@ethz.ch"

"""Brief: Trains a MLP predictor to classify Protein strings
"""

import numpy as np
import pandas as pd
import os
#import plaidml.keras
#plaidml.keras.install_backend()
import keras
import keras.backend as K
from PIL import Image
from keras.applications import densenet, inception_v3, mobilenet_v2,ResNet50
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Input, Dropout, Flatten, BatchNormalization, Activation
from keras.models import Model


from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Lambda
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

#functions:
def triplet_loss(y_true, y_pred):
        margin = K.constant(0.2)
        return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))
#
def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])
#
def l2Norm(x):
    return K.l2_normalize(x, axis=-1)
#
def euclidean_distance(vects):
    x, y = vects    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

#
def permutation(f):
    p = [0,2,1]
    y = []
    for i in range(0, int(f.shape[0]/2)):
        y = np.append(y, [1], axis=0)
    for i in range(int(f.shape[0]/2), f.shape[0]):
        f[i,:] = f[i,p]
        y = np.append(y, [0], axis=0)
    return f, y.astype('int')





#global vars:
h=50
w=50
dim=59515
dimtest=59544
def load_rgb_image(image_path,h,w):
    return np.array(Image.open(image_path).resize((h, w)))

#----Load Triplet Data--
# Note I added anchor, positive and negative to the head to have some easier data handling
df_train = pd.read_csv(os.path.join('..','data','train_triplets.txt'),delimiter=r"\s+")
df_test = pd.read_csv(os.path.join('..','data','test_triplets.txt'),delimiter=r"\s+")

print(df_train.shape)
print(df_test.shape)
#--------------------

#----Stack Image Matrix----
anchor =np.zeros((dim,h,w,3))
positive =np.zeros((dim,h,w,3))
negative =np.zeros((dim,h,w,3))

for n,val in enumerate(df_train.anchor[0:dim]):

    image_anchor = load_rgb_image(os.path.join('..','data','food',str(val).zfill(5))+'.jpg',h,w)
    image_anchor = image_anchor.astype("float32")
    # #image_anchor = image_anchor/255.
    image_anchor = keras.applications.resnet50.preprocess_input(image_anchor, data_format='channels_last')
    anchor[n] = image_anchor

print(anchor.shape)

for n,val in enumerate(df_train.positive[0:dim]):

    image_positive = load_rgb_image(os.path.join('..','data','food',str(val).zfill(5))+'.jpg',h,w)
    image_positive = image_positive.astype("float32")
    # #image_anchor = image_anchor/255.
    image_positive = keras.applications.resnet50.preprocess_input(image_positive, data_format='channels_last')
    positive[n] = image_positive

print(positive.shape)


for n,val in enumerate(df_train.negative[0:dim]):

    image_negative = load_rgb_image(os.path.join('..','data','food',str(val).zfill(5))+'.jpg',h,w)
    image_negative = image_negative.astype("float32")
    # #image_anchor = image_anchor/255.
    image_negative = keras.applications.resnet50.preprocess_input(image_negative, data_format='channels_last')
    negative[n] = image_negative

print(negative.shape)
#--------------------

#Train data:
Y_train = np.ones(dim)
print(Y_train)

#create neural net:
resnet_input = Input(shape=(h,w,3))
resnet_model = ResNet50(weights='imagenet', include_top = False, input_tensor=resnet_input)


for layer in resnet_model.layers:
    layer.trainable = False


net = resnet_model.output
net = Flatten(name='flatten')(net)
net = Dense(128, activation='relu', name='embed')(net)
net = Dense(128, activation='relu', name='embed2')(net)
net = Dense(128, activation='relu', name='embed3')(net)
net = Lambda(l2Norm, output_shape=[128])(net)

base_model = Model(resnet_model.input, net, name='resnet_model')

input_shape=(h,w,3)
input_anchor = Input(shape=input_shape, name='input_anchor')
input_positive = Input(shape=input_shape, name='input_pos')
input_negative = Input(shape=input_shape, name='input_neg')

net_anchor = base_model(input_anchor)
net_positive = base_model(input_positive)
net_negative = base_model(input_negative)

positive_dist = Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
negative_dist = Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])

stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists'
)([positive_dist, negative_dist])


model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')

model.compile(optimizer="rmsprop", loss=triplet_loss, metrics=[accuracy])

model.fit([anchor, positive, negative], Y_train, epochs=10,  batch_size=128, validation_split=0.2)

model.save('triplet_loss_resnet50.h5')


model = keras.models.load_model('triplet_loss_resnet50.h5',custom_objects={'triplet_loss': triplet_loss})


# predicting:

#----Stack Image Matrix----
anchor =np.zeros((dim,h,w,3))
positive =np.zeros((dim,h,w,3))
negative =np.zeros((dim,h,w,3))

for n,val in enumerate(df_test.anchor[0:dim]):

    image_anchor = load_rgb_image(os.path.join('..','data','food',str(val).zfill(5))+'.jpg',h,w)
    image_anchor = image_anchor.astype("float32")
    # #image_anchor = image_anchor/255.
    image_anchor = keras.applications.resnet50.preprocess_input(image_anchor, data_format='channels_last')
    anchor[n] = image_anchor

print(anchor.shape)

for n,val in enumerate(df_test.positive[0:dim]):

    image_positive = load_rgb_image(os.path.join('..','data','food',str(val).zfill(5))+'.jpg',h,w)
    image_positive = image_positive.astype("float32")
    # #image_anchor = image_anchor/255.
    image_positive = keras.applications.resnet50.preprocess_input(image_positive, data_format='channels_last')
    positive[n] = image_positive

print(positive.shape)


for n,val in enumerate(df_test.negative[0:dim]):

    image_negative = load_rgb_image(os.path.join('..','data','food',str(val).zfill(5))+'.jpg',h,w)
    image_negative = image_negative.astype("float32")
    # #image_anchor = image_anchor/255.
    image_negative = keras.applications.resnet50.preprocess_input(image_negative, data_format='channels_last')
    negative[n] = image_negative

print(negative.shape)


pred=model.predict([anchor, positive, negative])
print(pred)
outvec=np.zeros(dimtest)
for i,val in enumerate(pred):
    if val[0]<val[1]:
        outvec[i]=1
        print(val)
    else:
        outvec[i] = 0
        print(val)
print(outvec)
np.savetxt("task4.csv", outvec, fmt='%i',delimiter="\n")