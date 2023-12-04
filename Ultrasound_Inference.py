# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 01:48:57 2023

@author: Mohammed

"""
import keras.backend as K
from keras.models import load_model
import pickle 
import os 
import cv2 
import numpy as np 
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from tensorflow import keras
#%%
keras.utils.set_random_seed(812)

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
path=input("your model checkpoint")
model=load_model(path,custom_objects={'f1_metric':f1_metric})
with open(input("your testing sets as numpy"),'rb') as f:  # Python 3: open(..., 'rb')
    _,_,te1bb,te1mm,_,_ = pickle.load(f)
    
## These are samples names for benign class and malignant class 
## use them for inference 
benign=["benign "+t.split()[-1] for t in te1bb]
mal=["malignant "+t.split()[-1] for t in te1mm]




## path for dataset (denoised):
path='E:/us' 
ds1=[]
for bn in benign:
    img=cv2.imread(os.path.join(path,"benign",bn))

    img=cv2.resize(img,(224,224))
    ds1.append(img)

for mn in mal:
    img=cv2.imread(os.path.join(path,"malignant",mn))

    img=cv2.resize(img,(224,224))
    ds1.append(img)

ys1=[]
ts1=[]
lb1=[0]*len(te1bb)+[1]*len(te1mm)
ds1=np.array(ds1)
y_pred = model.predict(ds1,32)
ys1.append(y_pred)
ts1.append(lb1)
y_pred = np.where(y_pred>0.5, 1, 0)

confusion_matrix = metrics.confusion_matrix(y_true=lb1, y_pred=y_pred)  # shape=(12, 12)

print(confusion_matrix)



print(classification_report(lb1, y_pred))


 


## Original data path 
path='E:/Dataset_BUSI/Dataset_BUSI_with_GT'
ds1=[]
for bn in benign:
    img=cv2.imread(os.path.join(path,"benign",bn))

    img=cv2.resize(img,(224,224))
    ds1.append(img)

for mn in mal:
    img=cv2.imread(os.path.join(path,"malignant",mn))

    img=cv2.resize(img,(224,224))
    ds1.append(img)

ys1=[]
ts1=[]
lb1=[0]*len(te1bb)+[1]*len(te1mm)
ds1=np.array(ds1)
y_pred = model.predict(ds1,32)
ys1.append(y_pred)
ts1.append(lb1)
y_pred = np.where(y_pred>0.5, 1, 0)


confusion_matrix = metrics.confusion_matrix(y_true=lb1, y_pred=y_pred)  # shape=(12, 12)

print(confusion_matrix)



print(classification_report(lb1, y_pred))
