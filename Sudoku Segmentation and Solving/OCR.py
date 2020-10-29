from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as igd
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df

def get_data():
    mnist=fetch_openml('mnist_784',version=1)
    data=mnist["data"]
    target=mnist["target"]
    
    target=target.astype('int')
    data=data.astype('float32')

    x_train=data[:60000]
    y_train=target[:60000]
    xval=data[60000:]
    yval=target[60000:]
    
    x_train=x_train[(y_train!=0)]
    xval=xval[(yval!=0)]
    y_train=y_train[y_train!=0]
    yval=yval[yval!=0]

    y_train=y_train.reshape((-1,1))
    yval=yval.reshape((-1,1))

    x_train=x_train.reshape((len(x_train),28,28,1))
    x_train=x_train/255.

    xval=xval.reshape((len(xval),28,28,1))
    xval=xval/255.

    z=np.random.choice([0]*np.random.randint(10,30)+[255],(10000,28,28,1))
    z=z.astype('float32')
    z=z/255.

    z1=np.random.choice([0]*np.random.randint(10,30)+[255],(5000,28,28,1))
    z1=z1.astype('float32')
    z1=z1/255.

    zy=np.array([0]*10000,dtype='int').reshape((-1,1))
    zy1=np.array([0]*5000,dtype='int').reshape((-1,1))

    ext=pd.read_csv('ocrdata.csv').values
    ext=ext[:,1:]
    ext_x=ext[:,:-1].astype('float32').reshape((-1,28,28,1))
    ext_y=ext[:,-1].reshape((-1,1))
    ext_x=ext_x/255.
    ext_x=np.vstack([ext_x]*100)
    ext_y=np.vstack([ext_y]*100)

    x_train=np.concatenate((x_train,z,ext_x[:5500]))
    xval=np.concatenate((xval,z1,ext_x[5500:]))

    y_train=np.concatenate((y_train,zy,ext_y[:5500]))
    yval=np.concatenate((yval,zy1,ext_y[5500:]))

    ct=ColumnTransformer([('encoder',OneHotEncoder(),[0])])
    y_train=(ct.fit_transform(y_train)).toarray()
    yval=(ct.transform(yval)).toarray()
    return (x_train,y_train),(xval,yval)

def create_model():
    mod=tf.keras.Sequential()
    mod.add(tf.keras.layers.Input(shape=(28,28,1)))
    mod.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'))
    mod.add(tf.keras.layers.MaxPool2D((2,2),strides=2))
    mod.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'))
    mod.add(tf.keras.layers.MaxPool2D((2,2),strides=2))
    mod.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'))
    mod.add(tf.keras.layers.MaxPool2D((2,2),strides=2))
    mod.add(tf.keras.layers.Flatten())
    mod.add(tf.keras.layers.Dense(128,activation='relu'))
    mod.add(tf.keras.layers.Dropout(0.2))
    mod.add(tf.keras.layers.Dense(10,activation='softmax'))
    mod.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return mod

def train_model():
    model=create_model()
    (x_train,y_train),(xval,yval)=get_data()
    traingen=igd(rotation_range=20,shear_range=0.1)
    traingen.fit(x_train)
    train=traingen.flow(x_train,y_train,batch_size=32)
    valgen=igd(rotation_range=10,shear_range=0.1)
    valgen.fit(xval)
    val=valgen.flow(xval,yval,batch_size=32)
    callback_c=[tf.keras.callbacks.ModelCheckpoint('ocr.h5',monitor='val_loss',save_best_only=False,save_weights_only=False),
               tf.keras.callbacks.ModelCheckpoint('ocr2.h5',monitor='val_loss',save_best_only=False,save_weights_only=True)]
    history=model.fit(train,steps_per_epoch=len(x_train)//32,epochs=20,callbacks=callback_c,validation_data=val,validation_steps=len(xval)//32)


