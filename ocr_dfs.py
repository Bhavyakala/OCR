import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import sys

sys.setrecursionlimit(10**8)
path = 'D:/Coding_wo_cp/OCR/0325updated.task1train(626p)/X00016469612.jpg'
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
image = cv2.resize(image,(size,size))
origResize = cv2.resize(orig,(size,size))
ret,im = cv2.threshold(image,200,255,cv2.THRESH_BINARY_INV)
# edges = cv2.Canny(im)
while True:
    cv2.imshow("i",im)
    if cv2.waitKey(0) :
        break
cv2.destroyAllWindows()    
plt.imshow(im)

components = []
vis = [[False for j in range(size+1)]for i in range(size+1)]
def isSafe(i,j,n,m,vis) :
    return (i >= 0 and i < m and 
            j >= 0 and j < n and 
            not vis[i][j] and im[i][j]==255)

def dfs(r,c,vis,coordinatesList):
    rows = [-1,-1,-1, 1,1,1, 0,0]
    cols = [-1, 0, 1,-1,0,1,-1,1]
    vis[r][c]=True
    coordinatesList.append((r,c))
    for i in range(len(rows)) :
        cr = r + rows[i]
        cc = c + cols[i]
        if isSafe(cr,cc,size,size,vis):
            dfs(r+rows[i],c+cols[i],vis,coordinatesList)

def connectedComponents():
    count=0
    coordinatesList = []
    for i in range(size):
        for j in range(size):
            # print(i,j)
            if (not vis[i][j]) and im[i][j]==255:
                count+=1
                dfs(i,j,vis,coordinatesList)
                # if len(coordinatesList)>5 :
                components.append(coordinatesList)
                    # print(coordinatesList)
                coordinatesList = []   

connectedComponents()
# print("components")
# for i in range(len(components)):
#     print(components[i])

for i in range(len(components)) :
    xmin,_ = min(components[i],key = lambda t:t[0])
    xmax,_ = max(components[i],key = lambda t:t[0])
    _,ymin = min(components[i],key = lambda t:t[1])
    _,ymax = max(components[i],key = lambda t:t[1])
    # print(xmin,ymin,xmax,ymax)
    im3 = cv2.rectangle(origResize,(xmin,ymin),(xmax+2,ymax+2),(0,0,255),1)
# im3 = image[ymin:ymax-ymin,xmin:xmax-xmin]
while True:
    cv2.imshow("i",im3)
    if cv2.waitKey(0) :
        break
cv2.destroyAllWindows() 
plt.imshow(im3)
cv2.imwrite("im.jpeg",im3)
# components = np.array(components)