import cv2
import numpy as np
import skimage as sk
from skimage import transform
from skimage.transform import SimilarityTransform
import os
def rotate(img,angle,seed=1):
    if seed==0:
        return img
    dst=sk.transform.rotate(img.copy(),angle,preserve_range=True,mode='edge')
    return dst.astype(np.uint8)
def rotate_m(mask,angle,seed=1):
    if seed==0:
        return mask
    dst=sk.transform.rotate(mask.copy(),angle,order=0,preserve_range=True)
    return dst.astype(np.uint8)
def add_noise(img,seed=1):
    if seed==0:
        return img
    shp=img.shape
    noise=np.random.randint(0,20,(shp[0],shp[1],shp[2])).astype(np.uint8)
    dst=cv2.add(img,noise)
    return dst
def flip1(img,seed=1):
    if seed==0:
        return img
    dst=np.fliplr(img)
    return dst
def flip2(img,seed=1):
    if seed==0:
        return img
    dst=np.flipud(img)
    return dst
def shift(img,mask,seed=1):
    right=np.random.randint(-50,50)
    up=np.random.randint(-50,50)
    M=SimilarityTransform(translation=(right,up))
    dst1=sk.transform.warp(img,M,preserve_range=True,mode='edge')
    dst2=sk.transform.warp(mask,M,preserve_range=True,order=0)
    return dst1.astype(np.uint8),dst2.astype(np.uint8)
def create(n=200):
    ids=next(os.walk('sud_data/image/'))
    l=len(ids[2])
    k=1
    for i in range(1,n+1):
        ind=np.random.randint(0,l)
        ads=ids[2][ind]
        image=cv2.imread(ids[0]+ads)
        mask=cv2.imread('sud_data/mask/mask'+ads[3:-3]+'tif',0)
        check=np.random.randint(0,2,5)
        perm=np.random.permutation(range(5))
        dst1=image.copy()
        dst2=mask.copy()
        angle=np.random.randint(0,45)
        dst1=rotate(dst1,angle,check[perm[0]])
        dst2=rotate_m(dst2,angle,check[perm[0]])
        dst1=add_noise(dst1,check[perm[1]])
        dst1=flip1(dst1,check[perm[2]])
        dst2=flip1(dst2,check[perm[2]])
        dst1=flip2(dst1,check[perm[3]])
        dst2=flip2(dst2,check[perm[3]])
        dst1,dst2=shift(dst1,dst2,check[perm[4]])
        cv2.imwrite('sud_data/train/img/images/img'+str(k)+'.png',dst1)
        cv2.imwrite('sud_data/train/mask/masks/mask'+str(k)+'.png',dst2)
        k+=1
        
