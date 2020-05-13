import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import json

# path = 'D:/Coding_wo_cp/OCR/0325updated.task1train(626p)/X00016469619.jpg'
# path = 'D:/Coding_wo_cp/ML_CHALLENGE_HACKEREARTH/01e648028ad911ea/Dataset/Train_images/TR_2.jpg'
path = 'helloworld.png'
image = cv2.imread(path)
orig = image.copy() 
orig1 = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
h,w = orig.shape[:2]
print(orig.shape)


_,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
# edges = cv2.Canny(gray,50,150,apertureSize=3)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(im,contours,-1,(0,255,0),1)

boxes = []
new_contours = []
for i in range(len(contours)) :
    if hierarchy[0,i,3] == -1 : 
        x,y,w,h = cv2.boundingRect(contours[i])
        boxes.append((x,y,w,h))
        new_contours.append(contours[i])
        im1 = cv2.rectangle(orig1,(x-2,y-2),(x+w+2,y+h+2),(0,255,0),1)

# avg = sum(h for _,_,_,h in boxes)/len(boxes)

boxes.sort(key= lambda x :(x[1]+x[3]))
nb = []
temp = []
for i in range(1,(len(boxes))) :
    tolerance = 5
    x,y,w,h = boxes[i-1]
    x1,y1,w1,h1 = boxes[i]

    if ((y1+h1)-(y+h))<=10 :
        temp.append(boxes[i-1])
        if i==len(boxes)-1 :
            temp.append(boxes[i])
    else :
        if len(temp)>0 :
            nb.append(temp)
            temp=[]
    # print(y+h)
if len(temp)>0:
    nb.append(temp)


c=1
for i in range(len(nb)) :
    nb[i].sort(key = lambda x: x[0])
    for j in range(len(nb[i])) :
        x = nb[i][j][0]
        y = nb[i][j][1]
        im1 = cv2.putText(im1,str(c), (x,y),cv2.FONT_HERSHEY_COMPLEX,1,[125])
        c+=1


while True:
    cv2.imshow("i",im1)
    if cv2.waitKey(0) :
        break
cv2.destroyAllWindows()  
plt.imshow(im1)

mapp = pd.read_csv('emnist/emnist-balanced-mapping.txt',delimiter=' ', 
                   index_col=0,
                   header=None,
                   squeeze=True)
model = tf.keras.models.load_model('D:/Coding_wo_cp/OCR/model-3/model_5.h5')


c=1
for i in range(len(nb)) :
    for j in range(len(nb[i])) :

        plt.subplot(3,4,c)
        
        x,y,w,h=nb[i][j]
        im = orig[y-2:y+h+2,x-2:x+w+2]
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        im = np.pad(im,8,constant_values=255)
        im = cv2.resize(im,(28,28))
        im = cv2.bitwise_not(im)
        plt.imshow(im)

        im = np.expand_dims(im,axis=0)
        im = np.expand_dims(im,axis=3)
        pred = model.predict(im)
        idx = np.argmax(pred,axis=1)
        value = chr(mapp[idx])
        
        plt.title(value)
        c+=1



