import cv2
import numpy as np
import matplotlib.pyplot as plt
def show(img,name='img'):
    while(1):
        cv2.imshow(name,img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cv2.destroyAllWindows()
def show2(img,name='img'):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    while(1):
        cv2.imshow(name,img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cv2.destroyAllWindows()
def save(img,name):
    img=img.astype(np.uint8)
    cv2.imwrite(name,img)   
def hist_norm(gray):
    equ=cv2.equalizeHist(gray)
    return equ
def clahe_norm(gray):
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    cl1=clahe.apply(gray)
    return cl1
