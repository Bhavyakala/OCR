import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import cv2

from imutils.object_detection import non_max_suppression
# model = tf.keras.applications.vgg16.VGG16()

p = 'D:/Coding_wo_cp/OCR/0325updated.task1train(626p)/X00016469612.jpg'
image = cv2.imread(p)
orig = image.copy()
(H,W) = image.shape[:2]
rW = W / float(320)
rH = H / float(320)
image = cv2.resize(image,(320,320))
(H,W) = image.shape[:2]

layerNames = [
"feature_fusion/Conv_7/Sigmoid",
"feature_fusion/concat_3"]

net=cv2.dnn.readNet("D:/Coding_wo_cp/OCR/frozen_east_text_detection.pb")

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
(123.68, 116.78, 103.94), swapRB=True, crop=False)

net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

for y in range(0,numRows) :
    scoresData = scores[0,0,y]
    xData0 = geometry[0,0,y]
    xData1 = geometry[0,1,y]
    xData2 = geometry[0,2,y]
    xData3 = geometry[0,3,y]
    anglesData = geometry[0,4,y]
    for x in range(0, numCols) :
        if scoresData[x] < 0.5 :
            continue
        (offsetX, offsetY) = (x*4.0, y*4.0)
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

boxes = non_max_suppression(np.array(rects), probs=confidences) 
# loop over the bounding boxes

for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow("Text Detection", orig)

cv2.waitKey(0) 
# model.summary()