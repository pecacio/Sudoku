import tensorflow as tf
import cv2
import numpy as np
import datetime
from skimage.segmentation import clear_border
try:
    import cvtools as tt
except:
    tt=None
try:
    import sud_model as sm
except:
    sm=None
try:
    import OCR as ocr
except:
    ocr=None
try:
    import sudoku as sd
except:
    sd=None
if tt==None:
    raise ImportError('download cvtools, required for visualizing the solutions')
if sm==None:
    raise ImportError('download sud_model, required to get the sudoku segmentation model')
if ocr==None:
    raise ImportError('download OCR, required to get the digit recognition model')
if sd==None:
    raise ImportError('download sudoku, required to solve the sudoku')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
mod=0
ocrmod=0
def initialize():
    global mod,ocrmod
    print('[INFO] Loading the Sudoku Segmentation Model and the Digit Recognition model\n')
    mod=sm.u_net()
    try:
        mod.load_weights('sudoku_segmodel.h5')
    except:
        mod=None
        print('yo')
    if mod==None:
        raise RuntimeError('Download sudoku_segmodel.h5. Required for the segmentation model')
    ocrmod=ocr.create_model()
    try:
        ocrmod.load_weights('ocr_model.h5')
    except:
        ocrmod=None
    if ocrmod==None:
        raise RuntimeError('Download ocr_model.h5. Required for the digit recognition model') 
    print('[INFO] The program is ready for use\n') 
def get_mask(img):#img is of original size
    global mod
    img=cv2.resize(img,(256,256),cv2.INTER_CUBIC)
    x=img.astype('float32').reshape((1,256,256,3))
    x=x/255.0
    try:
        y1=mod.predict(x)
    except:
        y1=0
    if type(y1)==int:
        raise RuntimeError("Use the 'initialize' function before using the 'connect' function\n")
    t=np.where(y1>0.5,255,0)
    t=t.astype(np.uint8).reshape((256,256))
    return t
def crop_sudoku(img,mask,testing1=True,testing2=True):#img is of original size and mask is of size 256x256
    cont,hier=cv2.findContours(mask,cv2.RETR_TREE,2)
    ratio=(img.shape[0])/256
    cnts=sorted(cont,key=cv2.contourArea,reverse=True)
    outline=None
    for cnt in cnts:
        cnt=np.array(cnt,dtype='float32')
        cnt=cnt*ratio
        cnt=cnt.astype('int')
        epsilon=0.1*cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,epsilon,True)
        if len(approx)==4:
            outline=approx
            break
    if outline is None:
        return img
    if testing1:
        img1=cv2.drawContours(img.copy(),[outline],-1,(0,255,0),3)
        return img1
    outline=outline.reshape((4,2)).astype('float32')
    s=[x**2+y**2 for (x,y) in outline]
    ind1=s.index(min(s))
    ind4=s.index(max(s))
    l=outline.copy()
    l[[ind1,ind4],:]=0
    ind2=l[:,0].argmax()
    l[ind2,:]=0
    ind3=l[:,0].argmax()
    pts1=np.float32(outline[[ind1,ind2,ind3,ind4]])
    r,_,_=img.shape
    r=(r//9+1)*9
    pts2=np.float32([[0,0],[r+2,0],[0,r+2],[r+2,r+2]])
    M=cv2.getPerspectiveTransform(pts1,pts2)
    dst=cv2.warpPerspective(img,M,(r+2,r+2))
    if testing2:
        tt.show2(dst)
    return dst

def kernel(n=3):
    return np.ones((n,n),np.uint8)

def detect_and_extract_digits(img,testing=True):#rxr
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r,_,_=img.shape
    w0=int((r-2)/9)
    arr=np.zeros((9,9),dtype='int')
    mask=np.zeros((w0,w0),dtype=np.uint8)
    arr2=[]
    for i in range(9):
        for j in range(9):
            mask[:,:]=0
            x=2+j*w0
            y=2+i*w0
            roi=gray[y:y+w0,x:x+w0].copy()
            thresh=cv2.threshold(roi,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh=clear_border(thresh)
            filled=((thresh==255).sum())/(w0**2)
            if filled<0.03:
                arr[i,j]=-1
            else:
                cont,hier=cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)
                cnt=max(cont,key=cv2.contourArea)
                x,y,w,h=cv2.boundingRect(cnt)
                if (w*h/w0**2)<0.08:
                    arr[i,j]=-1
                else:
                    r1=(w0-h)//2
                    c1=(w0-w)//2
                    dig=thresh[y:y+h,x:x+w].copy()
                    mask[r1:r1+h,c1:c1+w]=dig
                    val=predict_digit(mask)
                    if val==0:
                        val=-1
                    arr[i,j]=val
            if testing:
                tt.show2(cv2.resize(mask,(28,28),cv2.INTER_CUBIC))
    return arr
def predict_digit(img):
    global ocr
    cpy=cv2.resize(img,(28,28),cv2.INTER_CUBIC)
    cpy=cpy.reshape((1,28,28,1)).astype('float32')
    cpy=cpy/255.0
    result=np.argmax(ocrmod.predict(cpy))
    return result
def connect(name,testing=False):
    start=datetime.datetime.now()
    img=cv2.imread(name,1)
    print('[INFO] Obtaining the segmentation mask of the Sudoku\n')
    mask=get_mask(img.copy())
    s=crop_sudoku(img.copy(),mask,False,testing)
    print('[INFO] Cropping the Sudoku Puzzle\n')
    arr=detect_and_extract_digits(s,testing)
    print('[INFO] Extracting and Detecting the digits\n')
    arr1=arr.copy()
    stop=datetime.datetime.now()
    print('[INFO] Detected within '+str((stop-start).total_seconds())+' seconds\n')
    start=datetime.datetime.now()
    print('[INFO] Solving the puzzle by depth-first-search algorithm\n')
    soln=sd.dfs(arr)
    stop=datetime.datetime.now()
    if type(soln)==int:
        print('Puzzle:\n')
        sd.print_board(arr1)
        if soln==-1:
            print('\nNo Solution Exists\n')
        elif soln==-2:
            print('\nInvalid Puzzle\n')
    else:
        print('[INFO] Solved Within '+str((stop-start).total_seconds())+' seconds\n')
        print('[INFO] Solution viewer is opened. (Puzzle on the left and solution on the right)\n')
        solution_viewer(soln,arr1)
        print('[INFO] Solution Viewer closed\n')
def solution_viewer(solved,board):
    image=cv2.imread('blank.png',1)
    s=image.shape[0]//9
    image1=image.copy()
    for i in range(9):
        for j in range(9):
            digit=solved[i,j]
            digit1=board[i,j]
            x=j*s
            y=i*s
            if digit1!=-1:
                dig=cv2.imread('digits/red/dig'+str(digit)+'.png',1)
                dig1=cv2.imread('digits/red/dig'+str(digit1)+'.png',1)
                dig1=cv2.resize(dig1,(s-20,s-20),cv2.INTER_CUBIC)
                image1[y+10:y+s-10:,x+10:x+s-10,:]=dig1.copy()
                dig1[:,:,:]=0
            else:
                dig=cv2.imread('digits/blue/dig'+str(digit)+'.png',1)
            dig=cv2.resize(dig,(s-20,s-20),cv2.INTER_CUBIC)
            image[y+10:y+s-10:,x+10:x+s-10,:]=dig.copy()
    cpy=np.hstack((np.zeros((image.shape[0],20,3),np.uint8),image1,np.zeros((image.shape[0],60,3),np.uint8),image,np.zeros((image.shape[0],20,3),np.uint8)))
    tt.show2(cpy,'Solved Puzzle')
