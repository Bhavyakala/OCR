import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import sys
import operator

sys.setrecursionlimit(10**8)
# path = 'D:/Coding_wo_cp/OCR/0325updated.task1train(626p)/X00016469619.jpg'
# path = 'D:/Coding_wo_cp/ML_CHALLENGE_HACKEREARTH/01e648028ad911ea/Dataset/Train_images/TR_2.jpg'
path = 'helloworld.png'
image = cv2.imread(path)
orig = image.copy() 
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
h,w = orig.shape[:2]
print(orig.shape)
size = 300
rH = size/h
rW = size/w
x = int(np.round(245*rW))
y = int(np.round(639*rH))
xmax = int(np.round(293*rW))
ymax = int(np.round(658*rH))
# image = cv2.resize(image,(size,size))
origResize = cv2.resize(orig,(size,size))
ret,im = cv2.threshold(image,200,255,cv2.THRESH_BINARY_INV)
# gray = cv2.bilateralFilter(im, 11, 17, 17)
# edged = cv2.Canny(im, 30, 200)
contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(im,contours,-1,(0,255,0),1)
# im1 = im[y-2:y+h+2,x-2:x+w+2]
# im1 = im[y-10:y+h+10,x-10:x+w+10]
boxes = []
for i in range(len(contours)) :
    if hierarchy[0,i,3] == -1:
        x,y,w,h = cv2.boundingRect(contours[i])
        boxes.append((x,y,w,h))
        im1 = cv2.rectangle(orig,(x-1,y-1),(x+w+1,y+h+1),(0,255,0),1)
# edges = cv2.Canny(im)

boxes.sort(key = lambda x:x[1])
# x,y,w,h=boxes[60]
# im1 = orig[y:y+h+1,x:x+w+1]
# im1 = orig[y-10:y+h+10,x-10:x+w+10]
# print(x,y,w,h)
while True:
    cv2.imshow("i",im1)
    if cv2.waitKey(0) :
        break
cv2.destroyAllWindows()    
plt.imshow(im1)