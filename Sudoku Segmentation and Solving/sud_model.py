import tensorflow as tf
import cv2
import numpy as np
import os
import json
import cvtools as tt
def parse_data():
    f=open('sud_data/raw/sud_labels.json')
    data=json.load(f)
    ids=next(os.walk('sud_data/raw/images/'))
    mask=np.zeros((256,256),np.uint8)
    h=1
    for i in ids[2]:
        mask[:,:]=0
        l=len(data[i]['regions'])
        for j in range(l):
            x_points=data[i]['regions'][str(j)]['shape_attributes']['all_points_x']
            y_points=data[i]['regions'][str(j)]['shape_attributes']['all_points_y']
            x,y=np.array(x_points,np.int32),np.array(y_points,np.int32)
            x,y=x.reshape((-1,1)),y.reshape((-1,1))
            z=np.concatenate((x,y),axis=1)
            z=z.reshape((-1,2,1))
            mask=cv2.fillPoly(mask,[z],(255,255,255),lineType=8)
        img=cv2.imread(ids[0]+i)
        res=cv2.bitwise_and(img,img,mask=mask)
        tt.show2(res)
        cv2.imwrite('sud_data/mask/mask'+str(h)+'.tif',mask)
        cv2.imwrite('sud_data/image/img'+str(h)+'.tif',img)
        h+=1
def get_data():
    X,Y=[],[]
    for i in range(1,1501):
        img=cv2.imread('sud_data/train/img/images/img'+str(i)+'.png')
        mask=cv2.imread('sud_data/train/mask/masks/mask'+str(i)+'.png',0)
        X.append(img.copy())
        Y.append(mask.copy())
    X=np.array(X)
    Y=np.array(Y)
    X=X.astype('float32').reshape((len(X),256,256,3))
    Y=Y.astype(np.bool).reshape((len(Y),256,256,1))
    X=X/255.0
    return X,Y
def seg_live():
    mod=u_net()
    mod.load_weights('colab_mod.h5')
    cap=cv2.VideoCapture(0)
    #cv2.namedWindow('col',cv2.WINDOW_NORMAL)
    while(1):
        ret,frame=cap.read()
        frame=cv2.resize(frame,(256,256))
        x=frame.astype('float32')
        x=x/255.0
        x=x.reshape((1,256,256,3))
        y1=mod.predict(x)
        t=np.where(y1>0.5,255,0)
        t=t.astype(np.uint8).reshape((256,256))
        col=cv2.cvtColor(t,cv2.COLOR_GRAY2BGR)
        res=cv2.bitwise_and(frame,frame,mask=t)
        cv2.imshow('col',res)
        cv2.imshow('mask',t)
        cv2.imshow('image',frame)
        k=cv2.waitKey(1) & 0xFF
        if k==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def check():
    mod=u_net()
    mod.load_weights('colab_mod.h5')
    for i in range(1,151):
        img=cv2.imread('sud_data/image/img'+str(i)+'.tif')
        x=img.astype('float32').reshape((1,256,256,3))
        x=x/255.0
        y1=mod.predict(x)
        t=np.where(y1>0.5,255,0)
        t=t.astype(np.uint8).reshape((256,256))
        res=cv2.bitwise_and(img,img,mask=t)
        while(1):
            cv2.imshow('img',img)
            cv2.imshow('res',res)
            cv2.imshow('mask',t)
            k=cv2.waitKey(1) & 0xFF
            if k==ord('q'):
                break
        cv2.destroyAllWindows()
def seg(img):
    mod=u_net()
    mod.load_weights('colab_mod.h5')
    x=img.astype('float32').reshape((1,256,256,3))
    x=x/255.0
    y1=mod.predict(x)
    t=np.where(y1>0.5,255,0)
    t=t.astype(np.uint8).reshape((256,256))
    res=cv2.bitwise_and(img,img,mask=t)
    col=cv2.cvtColor(t,cv2.COLOR_GRAY2BGR)
    dst=np.hstack((img,col,res))
    while(1):
        cv2.imshow('output',dst)
        k=cv2.waitKey(1) & 0xFF
        if k==ord('q'):
            break
    cv2.destroyAllWindows()
    return t
def u_net():
    inputs=tf.keras.layers.Input(shape=(None,None,3))

    #Contraction layer

    c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(inputs)
    c1=tf.keras.layers.Dropout(0.1)(c1)
    c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c1)
    p1=tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p1)
    c2=tf.keras.layers.Dropout(0.1)(c2)
    c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c2)
    p2=tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p2)
    c3=tf.keras.layers.Dropout(0.1)(c3)
    c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c3)
    p3=tf.keras.layers.MaxPooling2D((2,2))(c3)

    c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p3)
    c4=tf.keras.layers.Dropout(0.1)(c4)
    c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c4)
    p4=tf.keras.layers.MaxPooling2D((2,2))(c4)

    c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p4)
    c5=tf.keras.layers.Dropout(0.1)(c5)
    c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c5)
##    p5=tf.keras.layers.MaxPooling2D((2,2))(c5)
    
##    c6=tf.keras.layers.Conv2D(512,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p5)
##    c6=tf.keras.layers.Dropout(0.1)(c6)
##    c6=tf.keras.layers.Conv2D(512,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c6)
    ##p6=tf.keras.layers.MaxPooling2D((2,2))(c6)
    ##
    ##c7=tf.keras.layers.Conv2D(1024,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p6)
    ##c7=tf.keras.layers.Dropout(0.1)(c7)
    ##c7=tf.keras.layers.Conv2D(1024,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c7)

    #Expansion layer

    ##u8=tf.keras.layers.Conv2DTranspose(512,(2,2),strides=(2,2),padding='same')(c7)
    ##u8=tf.keras.layers.concatenate([u8,c6])
    ##c8=tf.keras.layers.Conv2D(512,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u8)
    ##c8=tf.keras.layers.Dropout(0.2)(c8)
    ##c8=tf.keras.layers.Conv2D(512,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c8)
    ##
##    u9=tf.keras.layers.Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(c6)
##    u9=tf.keras.layers.concatenate([u9,c5])
##    c9=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u9)
##    c9=tf.keras.layers.Dropout(0.2)(c9)
##    c9=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c9)

    u10=tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c5)
    u10=tf.keras.layers.concatenate([u10,c4])
    c10=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u10)
    c10=tf.keras.layers.Dropout(0.1)(c10)
    c10=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c10)

    u11=tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c10)
    u11=tf.keras.layers.concatenate([u11,c3])
    c11=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u11)
    c11=tf.keras.layers.Dropout(0.1)(c11)
    c11=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c11)

    u12=tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c11)
    u12=tf.keras.layers.concatenate([u12,c2])
    c12=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u12)
    c12=tf.keras.layers.Dropout(0.1)(c12)
    c12=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c12)

    u13=tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c12)
    u13=tf.keras.layers.concatenate([u13,c1])
    c13=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u13)
    c13=tf.keras.layers.Dropout(0.1)(c13)
    c13=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c13)

    outputs=tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(c13)

    model=tf.keras.Model(inputs=[inputs],outputs=[outputs])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    #model.summary()
    return model
##    callback_c=[tf.keras.callbacks.ModelCheckpoint(
##        'sudmodel2.h5',verbose=1,save_weights_only=False),tf.keras.callbacks.TensorBoard(log_dir='logs')]
##    model.fit(X,Y,batch_size=8,epochs=50,callbacks=callback_c,verbose=1,validation_split=0.1)
##    tf.keras.models.save_model(model,'sudoku_model2.h5')
##def detect(img):
##	global mod
##	x=img.astype('float32').reshape((1,256,256,3))
##	x=x/255.0
##	y1=mod.predict(x)
##	t=np.where(y1>0.5,255,0)
##	t=t.astype(np.uint8).reshape((256,256))
##	tt.show2(t)
##	cont,hier=cv2.findContours(t,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
##	cnts=sorted(cont,key=cv2.contourArea,reverse=True)
##	cnt=cnts[0]
##	e=0.1*cv2.arcLength(cnt,True)
##	approx=cv2.approxPolyDP(cnt,e,True)
##	if len(approx)==4:
##		print('True')
##		img=cv2.drawContours(img,[approx],-1,(0,0,255),2)
##	while(1):
##		cv2.imshow('image',img)
##		k=cv2.waitKey(1) & 0xFF
##		if k==ord('q'):
##			break
##	cv2.destroyAllWindows()
