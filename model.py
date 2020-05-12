import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth= True
tf.keras.backend.set_session(tf.Session(config=config))
import os
import cv2
import json


def model_arch(input_shape) :
    input_x = tf.keras.Input(input_shape)

    X = tf.keras.layers.Conv2D(64,(5,5),(2,2),'same',data_format='channels_last')(input_x)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.MaxPooling2D((2,2),padding='same')(X)
    X = tf.keras.layers.Dropout(0.25)(X)
    
    X = tf.keras.layers.Conv2D(128,(3,3),(2,2),'same')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.MaxPooling2D((2,2),padding='same')(X)
    X = tf.keras.layers.Dropout(0.25)(X)

    X = tf.keras.layers.Conv2D(256,(3,3),(2,2),'same')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.MaxPooling2D((2,2),padding='same')(X)
    X = tf.keras.layers.Dropout(0.5)(X)

    X = tf.keras.layers.Conv2D(512,(3,3),(2,2),'same')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.MaxPooling2D((2,2),padding='same')(X)
    X = tf.keras.layers.Dropout(0.25)(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(1024,activation='relu')(X)
    X = tf.keras.layers.Dense(512,activation='relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(128,activation='relu')(X)
    X = tf.keras.layers.Dense(36,activation='softmax')(X)

    model = tf.keras.Model(inputs=input_x, outputs=X)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def train() :
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory('D:/Coding_wo_cp/OCR/data/train', 
                                                        target_size=(128,128),
                                                        batch_size=128,
                                                        class_mode='categorical')
    valid_generator = datagen.flow_from_directory('D:/Coding_wo_cp/OCR/data/val', 
                                                        target_size=(128,128),
                                                        batch_size=32,
                                                        class_mode='categorical')
    test_generator = datagen.flow_from_directory('D:/Coding_wo_cp/OCR/data/test', 
                                                        target_size=(128,128),
                                                        batch_size=16,
                                                        class_mode='categorical')                                                    
    model = model_arch((128,128,3))
    return train_generator,valid_generator, test_generator, model

train_generator,valid_generator, test_generator, model = train()
model.summary()
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size


# with open('class_indices.json','w') as f:
#     json.dump(train_generator.class_indices,f)


model.fit_generator( generator=train_generator, 
                     steps_per_epoch=STEP_SIZE_TRAIN,
                     validation_data=valid_generator,
                     validation_steps=STEP_SIZE_VALID,
                     epochs=1)

model.save('model-1.h5')
model = tf.keras.models.load_model('D:\Coding_wo_cp\OCR\model-1.h5')

scores = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
print(scores)

im = cv2.imread('D:/Coding_wo_cp/OCR/by_class/by_class/0/train_30_00000.png')