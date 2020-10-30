import cv2
import numpy as np

#Functions to view images using opencv

def show(img,name='img'):#Showed in original size
    while(1):
        cv2.imshow(name,img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cv2.destroyAllWindows()
    
def show2(img,name='img'):#Image is fit to Window Size
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    while(1):
        cv2.imshow(name,img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cv2.destroyAllWindows()
    
def save(img,name):#Function to save a image
    img=img.astype(np.uint8)
    cv2.imwrite(name,img)
